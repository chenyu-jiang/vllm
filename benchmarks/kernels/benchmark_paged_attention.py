import argparse
import random
import time
import os
from dataclasses import dataclass

import torch

from vllm._C import ops

NUM_BLOCKS = 1024
PARTITION_SIZE = 512


@dataclass
class ProfileResult:
    batch_size: int
    block_size: int
    context_len: int
    dtype: str
    head_size: int
    num_kv_heads: int
    num_query_heads: int
    profile: bool
    seed: int
    use_alibi: bool
    version: str
    latency_us: float
    throughput_seqs: float
    throughput_kv_tokens: float
    throughput_kv_MB: float

    def to_csv_row(self):
        return ",".join([
            str(self.batch_size),
            str(self.block_size),
            str(self.context_len),
            self.dtype,
            str(self.head_size),
            str(self.num_kv_heads),
            str(self.num_query_heads),
            str(self.profile),
            str(self.seed),
            str(self.use_alibi),
            self.version,
            str(self.latency_us),
            str(self.throughput_seqs),
            str(self.throughput_kv_tokens),
            str(self.throughput_kv_MB),
        ])



@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    context_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
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

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

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

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    def run_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if version == "v1":
                ops.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )
            elif version == "v2":
                ops.paged_attention_v2(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")
    print(f"Throughput in Seqs: {num_seqs / latency:.3f} seqs/s")
    print(f"Throughput in KV Tokens: {num_seqs * context_len / latency:.3f} tokens/s")
    n_bytes_per_token = torch.tensor([], dtype=dtype).element_size() * head_size * num_kv_heads * 2
    print(f"Throughput in KV Bytes: {num_seqs * context_len / latency * n_bytes_per_token / 1000000:.3f} MB/s")
    if not os.path.exists("paged_attention_profile.csv"):
        with open("paged_attention_profile.csv", "w") as f:
            # write header
            f.write(",".join([
                "batch_size",
                "block_size",
                "context_len",
                "dtype",
                "head_size",
                "num_kv_heads",
                "num_query_heads",
                "profile",
                "seed",
                "use_alibi",
                "version",
                "latency_us",
                "throughput_seqs",
                "throughput_kv_tokens",
                "throughput_kv_MB",
            ]) + "\n")
    with open("paged_attention_profile.csv", "a") as f:
        f.write(ProfileResult(
            batch_size=num_seqs,
            block_size=block_size,
            context_len=context_len,
            dtype=str(dtype),
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            profile=do_profile,
            seed=seed,
            use_alibi=use_alibi,
            version=version,
            latency_us=latency * 1000000,
            throughput_seqs=num_seqs / latency,
            throughput_kv_tokens=num_seqs * context_len / latency,
            throughput_kv_MB=num_seqs * context_len / latency * n_bytes_per_token / 1000000,
        ).to_csv_row() + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }
    main(
        version=args.version,
        num_seqs=args.batch_size,
        context_len=args.context_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=dtype_to_torch_dtype[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
    )
