import argparse

import random

import torch
from vllm.transformers_utils.config import get_config
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.models.mixtral import MixtralAttention
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel,
)

NUM_BLOCKS = 10240

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_benchmark(batch_size, context_len, repeat: int, num_iters: int, warmup_iters: int, block_size=16) -> float:
    mixtral_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralAttention(mixtral_config.hidden_size,
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
    # Create the block tables.
    max_num_blocks_per_seq = (context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV cache.
    dtype = torch.bfloat16
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    head_size = mixtral_config.hidden_size // mixtral_config.num_attention_heads
    key_cache_shape = (NUM_BLOCKS, mixtral_config.num_key_value_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device="cuda")
    scale = float(1.0 / (head_size**0.5))
    key_cache.uniform_(-scale, scale)
    value_cache_shape = (NUM_BLOCKS, mixtral_config.num_key_value_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape,
                              dtype=dtype,
                              device="cuda")
    value_cache.uniform_(-scale, scale)
    kv_cache = (key_cache, value_cache)
    input_metadata = InputMetadata(
        is_prompt=False,
        slot_mapping=torch.randint(0, context_len, (batch_size, 1), dtype=torch.long, device="cuda"),
        max_context_len=context_len,
        context_lens=torch.tensor([context_len] * batch_size, dtype=torch.int, device="cuda"),
        block_tables=block_tables,
        use_cuda_graph=False,
    )
    # Benchmark.
    latencies = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # Warmup.
    for _ in range(warmup_iters):
        model(positions, dummy_input, kv_cache, input_metadata)
    for _ in range(repeat):
        start_ev.record()
        for _ in range(num_iters):
            model(positions, dummy_input, kv_cache, input_metadata)
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
        f = open(f"./mixtral_attention_tp{tp_size}{'_cudagraph' if use_cuda_graph else ''}.csv", "w")
        f.write("tp_size,batch_size,context_len,avg_latency\n")
    candidate_batch_size = [1, 2, 4] + [8 * i for i in range(1, 65)] + [640 + 128 * i for i in range(28)]
    candidate_context_len = [16 * i for i in range(1, 65)] + [1024 + 512 * i for i in range(7)]
    for batch_size in [x for x in candidate_batch_size]:
        for context_len in [x for x in candidate_context_len]:
            if use_cuda_graph:
                raise NotImplementedError
            else:
                avg_latency = run_benchmark(batch_size, context_len, repeat, num_iters, warmup_iters)
            if rank == 0:
                f.write(f"{tp_size},{batch_size},{context_len},{avg_latency}\n")
                f.flush()