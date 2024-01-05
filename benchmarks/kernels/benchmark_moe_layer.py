import argparse

import torch
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.mixtral import MixtralMoE, MixtralParallelism
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel
)

def run_benchmark_cudagraph(batch_size, repeat: int, num_iters: int,
                            warmup_iters: int, parallel_method: MixtralParallelism) -> float:
    mixtal_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralMoE(mixtal_config, parallel_method=parallel_method)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 4096).cuda().to(torch.bfloat16)
    # Warmup.
    for _ in range(warmup_iters):
        model(dummy_input)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(num_iters):
            model(dummy_input)
    # Benchmark.
    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start_ev.record()
        g.replay()
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev) / num_iters)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

def run_benchmark(batch_size, repeat: int, num_iters: int,
                  warmup_iters: int, parallel_method: MixtralParallelism) -> float:
    mixtal_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralMoE(mixtal_config, parallel_method=parallel_method)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 4096).cuda().to(torch.bfloat16)
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
    if tp_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=tp_size * dp_size,
        )
        initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                  data_parallel_size=dp_size)
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    print("Rank: {} initialized.".format(rank))
    if rank == 0:
        f = open(f"./mixtral_moe_tp{tp_size}_dp{dp_size}_{args.parallel_method}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("tp_size,dp_size,batch_size,avg_latency,parallel_method\n")
    parallel_method = MixtralParallelism.TENSOR_EXPERT_PARALLEL if args.parallel_method == "TP_EP" else MixtralParallelism.DATA_EXPERT_PARALLEL
    for batch_size in [1, 2, 4] + [8 * i for i in range(1, 65)] + [640, 768, 896, 1024]:
        if use_cuda_graph:
            avg_latency = run_benchmark_cudagraph(batch_size, repeat, num_iters, warmup_iters, parallel_method)
        else:
            avg_latency = run_benchmark(batch_size, repeat, num_iters, warmup_iters, parallel_method)
        if rank == 0:
            f.write(f"{tp_size},{dp_size},{batch_size},{avg_latency},{args.parallel_method}\n")
            f.flush()