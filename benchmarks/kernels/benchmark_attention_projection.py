import argparse
from typing import Optional

import torch
from torch import nn

from vllm.transformers_utils.config import get_config
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope

class MixtralAttentionProjectionOnly(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # dummy attn_out (slice by tp_size)
        attn_output = hidden_states[:, :, :self.q_size]
        output, _ = self.o_proj(attn_output)
        return output, q, k, v


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_benchmark(batch_size, repeat: int, num_iters: int, warmup_iters: int) -> float:
    mixtral_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralAttentionProjectionOnly(mixtral_config.hidden_size,
                             mixtral_config.num_attention_heads,
                             mixtral_config.num_key_value_heads,
                             mixtral_config.max_position_embeddings,
                             mixtral_config.rope_theta,
                             sliding_window=mixtral_config.sliding_window)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    set_seed(42)
    dummy_input = torch.randn(batch_size, 1, 4096).cuda().to(torch.bfloat16)
    positions = torch.randint(0, 4096, (batch_size, 1), dtype=torch.long, device="cuda")

    # Benchmark.
    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # Warmup.
    for _ in range(warmup_iters):
        model(positions, dummy_input)
    for _ in range(repeat):
        start_ev.record()
        for _ in range(num_iters):
            model(positions, dummy_input)
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev) / num_iters)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--use-cuda-graph", action="store_true")
    args = parser.parse_args()

    repeat = 20
    num_iters = 100
    warmup_iters = 20
    tp_size = args.tp_size
    use_cuda_graph = args.use_cuda_graph
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=tp_size,
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                data_parallel_size=1)
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    if rank == 0:
        f = open(f"./mixtral_attention_projection_tp{tp_size}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("tp_size,batch_size,avg_latency\n")
    candidate_batch_size = [1, 2, 4] + [8 * i for i in range(1, 65)] + [640 + 128 * i for i in range(28)]
    for batch_size in [x for x in candidate_batch_size]:
        if use_cuda_graph:
            raise NotImplementedError
        else:
            avg_latency = run_benchmark(batch_size, repeat, num_iters, warmup_iters)
        if rank == 0:
            f.write(f"{tp_size},{batch_size},{avg_latency}\n")
            f.flush()