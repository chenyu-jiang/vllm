import argparse
import os
import random
import time
from typing import List, Union

import torch

from vllm._C import ops
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata

NUM_BLOCKS = 10240
PARTITION_SIZE = 512
_PAD_SLOT_ID = -1

def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))

def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x,
                        dtype=dtype,
                        device=device,
                        pin_memory=pin_memory and str(device) == "cpu")

@torch.inference_mode()
def run_benchmark(
    num_seqs: int,
    context_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    sliding_window: int,
    do_profile: bool,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    key = torch.empty(num_seqs,
                    num_kv_heads,
                    head_size,
                    dtype=dtype,
                    device="cuda")
    key.uniform_(-scale, scale)

    value = torch.empty(num_seqs,
                        num_kv_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    value.uniform_(-scale, scale)
                

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    model = PagedAttention(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window)

    context_lens = [context_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device="cuda")
    key_cache.uniform_(-scale, scale)
    value_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape,
                              dtype=dtype,
                              device="cuda")
    value_cache.uniform_(-scale, scale)

    # Create the input metadata.
    slot_mapping = [
        [random.randint(0, NUM_BLOCKS * block_size - 1)] for _ in range(num_seqs)
    ]
    context_lens = [context_len] * num_seqs
    slot_mapping = _make_tensor_with_pad(slot_mapping,
                                        max_len=1,
                                        pad=_PAD_SLOT_ID,
                                        dtype=torch.long,
                                        device="cuda")
    context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device="cuda")
    input_metadata = InputMetadata(
        is_prompt=False,
        slot_mapping=slot_mapping,
        max_context_len=max_context_len,
        context_lens=context_lens,
        block_tables=block_tables,
        use_cuda_graph=False,
    )

    def run_bench_helper(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            model(query, key, value, key_cache, value_cache, input_metadata)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    run_bench_helper(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_bench_helper(num_iters=1, profile=True)
    else:
        latency = run_bench_helper(num_iters=100, profile=False)
    return latency * 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--sliding-window", type=int, default=4096)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }
    for batch_size in [1, 2, 4] + [8 * i for i in range(1, 65)] + [640 + 128 * i for i in range(12)]:
        for per_seq_context_len in [16 * i for i in range(1, 33)] + [128 * i for i in range(5, 33)]:
            latency = run_benchmark(
                    num_seqs=batch_size,
                    context_len=per_seq_context_len,
                    num_query_heads=args.num_query_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_size=args.head_size,
                    block_size=args.block_size,
                    use_alibi=args.use_alibi,
                    dtype=dtype_to_torch_dtype[args.dtype],
                    seed=args.seed,
                    sliding_window=args.sliding_window,
                    do_profile=args.profile,
                )
            fn = "./mixtral_paged_attention.csv"
            if not os.path.exists(fn):
                with open(fn, "w") as f:
                    f.write("batch_size,total_context_len,latency\n")
            with open(fn, "a") as f:
                f.write(f"{batch_size},{per_seq_context_len * batch_size},{latency}\n")
