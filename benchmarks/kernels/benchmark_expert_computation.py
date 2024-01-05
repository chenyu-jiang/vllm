from typing import Optional
import time
import argparse

import torch
from torch import nn

import numpy as np

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce
)

class MixtralParallelMoEWithoutGate(nn.Module):

    def __init__(
        self, n_experts: int, hidden_size: int, intermediate_size: int, unified_reduce: bool = False
    ):
        super().__init__()
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = n_experts
        self.unified_reduce = unified_reduce

        self.experts = nn.ModuleList([
            MixtralParallelMLP(hidden_size, intermediate_size, reduce_results=not unified_reduce)
            for _ in range(n_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_dim = hidden_states.shape

        final_hidden_states = None
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            current_hidden_states = expert_layer(hidden_states)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        if self.unified_reduce:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(batch_size, hidden_dim)

class MixtralParallelMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Optional[str] = "silu",
        linear_method: Optional[LinearMethodBase] = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           reduce_results=reduce_results,
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

def run_benchmark_cudagraph(n_experts, batch_size, repeat: int, num_iters: int, warmup_iters: int, unified_reduce: bool) -> float:
    model = MixtralParallelMoEWithoutGate(n_experts, hidden_size=4096, intermediate_size=14336, unified_reduce=unified_reduce)
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

def run_benchmark(n_experts, batch_size, repeat: int, num_iters: int, warmup_iters: int, unified_reduce: bool) -> float:
    model = MixtralParallelMoEWithoutGate(n_experts, hidden_size=4096, intermediate_size=14336, unified_reduce=unified_reduce)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--use-cuda-graph", action="store_true")
    args = parser.parse_args()

    repeat = 20
    num_iters = 100
    warmup_iters = 20
    tp_size = args.tp_size
    use_cuda_graph = args.use_cuda_graph
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
        f.write("n_experts,tp_size,batch_size,avg_latency,unified_reduce\n")
    for n_experts in [1, 2, 4, 8]:
        for unified_reduce in [False, True]:
            for batch_size in [1, 2, 4] + [8 * i for i in range(1, 65)] + [640, 768, 896, 1024]:
                if use_cuda_graph:
                    avg_latency = run_benchmark_cudagraph(n_experts, batch_size, repeat, num_iters, warmup_iters, unified_reduce)
                else:
                    avg_latency = run_benchmark(n_experts, batch_size, repeat, num_iters, warmup_iters, unified_reduce)
                if tp_rank == 0:
                    f.write(f"{n_experts},{tp_size},{batch_size},{avg_latency},{unified_reduce}\n")
                    f.flush()