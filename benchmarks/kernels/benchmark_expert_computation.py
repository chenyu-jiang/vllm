from typing import Optional
import time

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel
)

class MixtralParallelMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Optional[str] = "silu",
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states

def run_benchmark_cudagraph(batch_size, repeat: int, num_iters: int, warmup_iters: int, tp_size: int) -> float:
    if tp_size > 1:
        model = MixtralParallelMLP(hidden_size=4096, intermediate_size=14336)
    else:
        model = MixtralMLP(num_experts=1, hidden_size=4096, intermediate_size=14336)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    dummy_input = torch.randn(batch_size, 4096).cuda().to(torch.bfloat16)
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

def run_benchmark(batch_size, repeat: int, num_iters: int, warmup_iters: int, tp_size: int) -> float:
    if tp_size > 1:
        model = MixtralParallelMLP(hidden_size=4096, intermediate_size=14336)
    else:
        model = MixtralMLP(num_experts=1, hidden_size=4096, intermediate_size=14336)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    dummy_input = torch.randn(batch_size, 4096).cuda().to(torch.bfloat16)
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
    repeat = 20
    num_iters = 100
    warmup_iters = 20
    tp_size = 2
    use_cuda_graph = True
    if tp_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=tp_size,
        )
        initialize_model_parallel(tp_size)
        tp_rank = torch.distributed.get_rank()
        torch.cuda.set_device(tp_rank)
    else:
        tp_rank = 0
    if tp_rank == 0:
        f = open(f"./expert_computation_tp{tp_size}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("tp_size,batch_size,avg_latency\n")
    for batch_size in [1, 2, 4, 8, 16, 32, 40, 48, 56, 64, 128, 256, 384, 512, 1024]:
        if use_cuda_graph:
            avg_latency = run_benchmark_cudagraph(batch_size, repeat, num_iters, warmup_iters, tp_size)
        else:
            avg_latency = run_benchmark(batch_size, repeat, num_iters, warmup_iters, tp_size)
        if tp_rank == 0:
            f.write(f"{tp_size},{batch_size},{avg_latency}\n")
            f.flush()