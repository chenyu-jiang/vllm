import torch
import torch.distributed as dist

def AllToAll(tensor, group=None, async_op=False):
    """AllToAll"""
    if group is None:
        group = dist.group.WORLD
    output = torch.empty_like(tensor)
    return dist.all_to_all_single(output, tensor, group=group, async_op=async_op)


def run_benchmark(batch_size, repeat: int, num_iters: int, warmup_iters: int) -> float:
    dummy_input = torch.randn(batch_size, 4096).cuda().to(torch.bfloat16)
    # Benchmark.
    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # Warmup.
    for _ in range(warmup_iters):
        AllToAll(dummy_input)
    for _ in range(repeat):
        start_ev.record()
        for _ in range(num_iters):
            AllToAll(dummy_input)
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev) / num_iters)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

if __name__ == "__main__":
    repeat = 20
    num_iters = 100
    warmup_iters = 20
    world_size = 4
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
    )
    rank = torch.distributed.get_rank()
    use_cuda_graph = False
    torch.cuda.set_device(rank)
    if rank == 0:
        f = open(f"./alltoall_ws{world_size}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("world_size,batch_size,avg_latency\n")
    for batch_size in [1, 2, 4, 8, 16, 32, 40, 48, 56, 64, 128, 256, 384, 512, 1024]:
        if use_cuda_graph:
            raise NotImplementedError
            # avg_latency = run_benchmark_cudagraph(batch_size, repeat, num_iters, warmup_iters, tp_size)
        else:
            avg_latency = run_benchmark(batch_size, repeat, num_iters, warmup_iters)
        if rank == 0:
            f.write(f"{world_size},{batch_size},{avg_latency}\n")
            f.flush()