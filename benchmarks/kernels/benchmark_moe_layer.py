import argparse

import torch
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.mixtral import MixtralMoE, MixtralParallelism
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel,
    get_data_parallel_rank,
)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

def load_weights_first_layer_dp(model, model_name):
    params_dict = dict(model.named_parameters())
    stacked_params_mapping = [
        ("w1_w3", "w1", 0),
        ("w1_w3", "w3", 1),
    ]
    for name, loaded_weight in hf_model_weights_iterator(model_name, fall_back_to_pt=False):
        if name.startswith("model.layers.1.block_sparse_moe."):
            name = name[len("model.layers.1.block_sparse_moe."):]
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ("experts." in name and name not in params_dict):
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

def load_weights_first_layer_tp(model, model_name):
    params_dict = dict(model.named_parameters())
    for name, loaded_weight in hf_model_weights_iterator(model_name, fall_back_to_pt=False):
        if name.startswith("model.layers.1.block_sparse_moe."):
            name = name[len("model.layers.1.block_sparse_moe."):]
            if ("experts." in name and name not in params_dict):
                    continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_benchmark(batch_size, repeat: int, num_iters: int,
                  warmup_iters: int, parallel_method: MixtralParallelism) -> float:
    mixtal_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralMoE(mixtal_config, parallel_method=parallel_method)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    dp_rank = get_data_parallel_rank()
    if parallel_method == MixtralParallelism.TENSOR_EXPERT_PARALLEL:
        load_weights_first_layer_tp(model, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    else:
        load_weights_first_layer_dp(model, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    set_seed(42)
    dummy_input = torch.randn(batch_size, 1, 4096).cuda().to(torch.bfloat16)
    if parallel_method == MixtralParallelism.DATA_EXPERT_PARALLEL:
        # split input across data parallel dimension
        assert batch_size % dp_size == 0
        dummy_input = dummy_input[dp_rank * batch_size // dp_size: (dp_rank + 1) * batch_size // dp_size]
    # Benchmark.
    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # Warmup.
    for _ in range(warmup_iters):
        model(dummy_input)
    for _ in range(repeat):
        start_ev.record()
        for _ in range(num_iters):
            model(dummy_input)
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev) / num_iters)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--parallel-method", type=str, choices=["TP_EP", "DP_TP_EP"], default="TP_EP")
    args = parser.parse_args()

    repeat = 20
    num_iters = 100
    warmup_iters = 20
    tp_size = args.tp_size
    dp_size = args.dp_size
    use_cuda_graph = args.use_cuda_graph
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=tp_size * dp_size,
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                data_parallel_size=dp_size)
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    print("Rank: {} initialized.".format(rank))
    if rank == 0:
        f = open(f"./mixtral_moe_tp{tp_size}_dp{dp_size}_{args.parallel_method}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("tp_size,dp_size,batch_size,avg_latency,parallel_method\n")
    parallel_method = MixtralParallelism.TENSOR_EXPERT_PARALLEL if args.parallel_method == "TP_EP" else MixtralParallelism.DATA_EXPERT_PARALLEL
    candidate_batch_size = [1, 2, 4] + [8 * i for i in range(1, 65)] + [640, 768, 896, 1024]
    for batch_size in [x for x in candidate_batch_size if (x >= dp_size and x % dp_size == 0)]:
        if use_cuda_graph:
            raise NotImplementedError
        else:
            if rank == 0:
                print(f"Running benchmark for batch size {batch_size}...")
            avg_latency = run_benchmark(batch_size, repeat, num_iters, warmup_iters, parallel_method)
        if rank == 0:
            f.write(f"{tp_size},{dp_size},{batch_size},{avg_latency},{args.parallel_method}\n")
            f.flush()