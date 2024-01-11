import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MixtralConfig
from vllm.transformers_utils.config import get_config

from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)

class MixtralGateOnlyMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
    ):
        super().__init__()
        self.config = config
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts


def run_benchmark(config: MixtralConfig, batch_size, repeat: int, num_iters: int, warmup_iters: int) -> float:
    model = MixtralGateOnlyMoE(config=config)
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
    args = parser.parse_args()

    repeat = 20
    num_iters = 100
    warmup_iters = 20
    mixtral_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    with open(f"./mixtral_gate.csv", "w") as f:
        f.write("batch_size,avg_latency\n")
        for batch_size in tqdm([1, 2, 4] + [8 * i for i in range(1, 65)] + [640 + 128 * i for i in range(28)]):
            avg_latency = run_benchmark(mixtral_config, batch_size, repeat, num_iters, warmup_iters)
            f.write(f"{batch_size},{avg_latency}\n")
            f.flush()