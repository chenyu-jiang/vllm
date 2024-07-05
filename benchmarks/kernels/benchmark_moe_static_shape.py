import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

def get_config_file_extended_name(E: int, N: int, dtype: Optional[str], num_shared_experts: int = 0) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    if num_shared_experts == 0:
        return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"
    else:
        return f"E={E},N={N},ExpShared={num_shared_experts},device_name={device_name}{dtype_selector}.json"

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
    num_shared_experts: int = 0,
    num_iters: int = 100,
) -> float:
    init_dtype = torch.float16 if use_fp8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts + num_shared_experts,
                     shard_intermediate_size,
                     hidden_size,
                     dtype=init_dtype)
    w2 = torch.randn(num_experts + num_shared_experts,
                     hidden_size,
                     shard_intermediate_size // 2,
                     dtype=init_dtype)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_fp8:
        w1_scale = torch.randn(num_experts + num_shared_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts + num_shared_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

    topk_weights_buffer = torch.empty(num_tokens, topk + num_shared_experts, dtype=dtype)
    topk_indices_buffer = torch.empty(num_tokens, topk + num_shared_experts, dtype=torch.int32)

    def prepare(i: int):
        topk_weights, topk_ids = fused_topk(x, gating_output[i], topk, True)
        if num_shared_experts:
            topk_weights = torch.cat([topk_weights, torch.ones(num_shared_experts, dtype=dtype).expand(num_tokens, -1)], dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = torch.cat([topk_ids + num_shared_experts, torch.arange(0, num_shared_experts, 1).expand(num_tokens, -1)], dim=-1)
        topk_weights_buffer.copy_(topk_weights)
        topk_indices_buffer.copy_(topk_ids)

    def run():
        fused_experts(
            x,
            w1,
            w2,
            topk_weights_buffer,
            topk_indices_buffer,
            inplace=True,
            override_config=config,
            use_fp8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128, 256]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append({
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            })
    return configs


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

    def get_moe_configs_extended(self, E: int, N: int,
                    dtype: Optional[str], num_shared_experts: int = 0) -> Optional[Dict[int, Any]]:
        """
        Return optimized configurations for the fused MoE kernel.

        The return value will be a dictionary that maps an irregular grid of
        batch sizes to configurations of the fused_moe kernel. To evaluate the
        kernel on a given batch size bs, the closest batch size in the grid should
        be picked and the associated configuration chosen to invoke the kernel.
        """

        # First look up if an optimized configuration is available in the configs
        # directory
        json_file_name = get_config_file_extended_name(E, N, dtype, num_shared_experts)

        config_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                print(f"Using configuration from {config_file_path} for MoE layer.")
                # If a configuration has been found, return it
                return {int(key): val for key, val in json.load(f).items()}

        # If no optimized configuration is available, we will use the default
        # configuration
        return None

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8: bool,
        num_shared_experts=0,
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(self.seed)

        dtype_str = "float8" if use_fp8 else None
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        op_config = self.get_moe_configs_extended(num_experts, shard_intermediate_size // 2,
                                    dtype_str, num_shared_experts=num_shared_experts)
        if op_config is None:
            config = get_default_config(num_tokens, num_experts,
                                        shard_intermediate_size, hidden_size,
                                        topk, dtype_str)
        else:
            config = op_config[min(op_config.keys(),
                                   key=lambda x: abs(x - num_tokens))]
            print(f"Using precomputed config: {config}")
        kernel_time = benchmark_config(config, num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8, num_shared_experts=num_shared_experts)
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8: bool,
        num_shared_experts: int,
        search_space: List[BenchmarkConfig],
    ) -> BenchmarkConfig:
        best_config = None
        best_time = float("inf")
        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(config,
                                               num_tokens,
                                               num_experts,
                                               shard_intermediate_size,
                                               hidden_size,
                                               topk,
                                               dtype,
                                               use_fp8,
                                               num_shared_experts=num_shared_experts,
                                               num_iters=10)
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config
        now = datetime.now()
        print(f"[{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
    num_shared_experts: int = 0,
) -> None:
    dtype_str = "float8" if use_fp8 else None
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_extended_name(num_experts, shard_intermediate_size // 2,
                                    dtype_str, num_shared_experts)
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    E = args.num_experts
    topk = args.topk
    shard_intermediate_size = args.shard_intermediate_size

    hidden_size = args.hidden_size
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    use_fp8 = args.dtype == "fp8"
    num_shared_experts = args.num_shared_experts

    if args.batch_size is None:
        batch_sizes = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
            2048, 3072, 4096
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        search_space = get_configs_compute_bound()
        print(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        configs = _distribute(
            "tune", [(batch_size, E, shard_intermediate_size, hidden_size,
                      topk, dtype, use_fp8, num_shared_experts, search_space)
                     for batch_size in batch_sizes])
        best_configs = {
            M: sort_config(config)
            for M, config in zip(batch_sizes, configs)
        }
        save_configs(best_configs, E, shard_intermediate_size, hidden_size,
                     topk, dtype, use_fp8, num_shared_experts=num_shared_experts)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute("benchmark",
                              [(batch_size, E, shard_intermediate_size,
                                hidden_size, topk, dtype, use_fp8, num_shared_experts)
                               for batch_size in batch_sizes])

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--num-experts", "-e", type=int, default=20)
    parser.add_argument("--shard-intermediate-size", "-s", type=int, default=1536)
    parser.add_argument("--num-shared-experts", "-nse", type=int, default=2)
    parser.add_argument("--topk", "-k", type=int, default=6)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["bfloat16", "fp8"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
