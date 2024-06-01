import argparse
import random
import time
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, create_kv_caches_with_random, make_tensor_with_pad

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm.attention.backends.flash_attn import FlashAttentionBackend

from vllm.distributed import divide

from vllm.model_executor.utils import set_weight_attrs

from vllm.sequence import SequenceGroupMetadata, SequenceData, SamplingParams

from vllm.worker.model_runner import BatchType, PrepareDecodeMetadata

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ReplicatedLinear,
                                               UnquantizedLinearMethod)

PARTITION_SIZE = 512

def concat_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_cache_last_dim = value_cache.shape[-1]
    key_cache = key_cache.view(key_cache.shape[0],
                               key_cache.shape[1],
                               -1,
                               value_cache_last_dim)
    return torch.concatenate([key_cache, value_cache], dim=0)

def _prepare_decode(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    sliding_window: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    block_size: int = 16,
) -> PrepareDecodeMetadata:
    input_tokens: List[int] = []
    input_positions: List[int] = []
    slot_mapping: List[int] = []
    context_lens: List[int] = []
    block_tables: List[List[int]] = []
    lora_index_mapping: List[int] = []
    lora_prompt_mapping: List[int] = []
    lora_requests = set()

    if len(seq_group_metadata_list) == 0:
        return PrepareDecodeMetadata.empty()

    for seq_group_metadata in seq_group_metadata_list:
        assert not seq_group_metadata.is_prompt
        assert seq_group_metadata.token_chunk_size == 1

        seq_ids = list(seq_group_metadata.seq_data.keys())
        lora_id = seq_group_metadata.lora_int_id

        if lora_id > 0:
            lora_requests.add(seq_group_metadata.lora_request)

        for seq_id in seq_ids:
            seq_data = seq_group_metadata.seq_data[seq_id]
            generation_token = seq_data.get_last_token_id()
            input_tokens.append(generation_token)

            seq_len = seq_data.get_len()
            position = seq_len - 1
            input_positions.append(position)

            context_len = seq_len if sliding_window is None else min(
                seq_len, sliding_window)
            context_lens.append(context_len)

            block_table = seq_group_metadata.block_tables[seq_id]
            block_number = block_table[position // block_size]
            block_offset = position % block_size
            slot = block_number * block_size + block_offset
            slot_mapping.append(slot)
            lora_index_mapping.append(lora_id)
            lora_prompt_mapping.append(lora_id)

            if sliding_window is not None:
                sliding_window_blocks = (sliding_window //
                                            block_size)
                block_table = block_table[-sliding_window_blocks:]
            block_tables.append(block_table)

    # vLLM uses cuda graph only for decoding requests.
    # See `capture_model` API for more details.
    # For decoding requests, batch_size == input_tokens.
    batch_size = len(input_tokens)
    max_context_len = max(context_lens)

    context_lens = torch.tensor(context_lens,
                                dtype=torch.int,
                                device=device)


    max_block_table_len = max(
        len(block_table) for block_table in block_tables)
    block_tables = make_tensor_with_pad(
        block_tables,
        max_len=max_block_table_len,
        pad=0,
        dtype=torch.int,
        device=device,
    )

    attn_metadata = FlashAttentionBackend.make_metadata(
        is_prompt=False,
        prompt_lens=None,
        prompt_lens_tensor=None,
        max_subquery_len=None,
        max_context_len=max_context_len,
        max_prompt_len=None,
        subquery_start_loc=None,
        seq_start_loc=None,
        context_lens=context_lens,
        block_tables=block_tables,
        use_cuda_graph=False,
    )
    return PrepareDecodeMetadata(
        input_tokens=input_tokens,
        input_positions=input_positions,
        attn_metadata=attn_metadata,
        lora_index_mapping=lora_index_mapping,
        lora_prompt_mapping=lora_prompt_mapping,
        lora_requests=lora_requests,
        slot_mapping=slot_mapping,
    )

def prepare_decode(batch_size, context_len, block_size, num_kv_heads, head_size, kv_cache_dtype, dtype, device):
    prompt_lens = []
    seq_group_metadata_list = []

    n_blocks_per_seq = (context_len + 1 + block_size - 1) // block_size
    n_blocks = n_blocks_per_seq * batch_size

    for i in range(batch_size):
        # make sure all tokens fit into one block
        prompt_len = context_len
        prompt_lens.append(prompt_len)
        seq_data = list(range(prompt_len))
        seq_data = SequenceData(seq_data)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables = {0: [i * n_blocks_per_seq + j for j in range(n_blocks_per_seq)]},
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)

    (
        decode_input_tokens,
        decode_input_positions,
        decode_attn_metadata,
        decode_lora_index_mapping,
        decode_lora_prompt_mapping,
        decode_lora_requests,
        decode_slot_mapping,
    ) = _prepare_decode(
        seq_group_metadata_list,
        sliding_window=None,
        block_size=block_size,
        device=device,
    )

    input_positions = torch.tensor(decode_input_positions,
                                    dtype=torch.long,
                                    device=device)
    slot_mapping = torch.tensor(decode_slot_mapping,
                                dtype=torch.long,
                                device=device)

    attn_metadata = AttentionMetadata(
        num_prefills=0,
        slot_mapping=slot_mapping,
        num_prefill_tokens=0,
        num_decode_tokens=batch_size,
        prefill_metadata=None,
        decode_metadata=decode_attn_metadata,
        kv_cache_dtype=kv_cache_dtype,
    )

    return n_blocks, input_positions, attn_metadata

def prepare_attn_metadata(num_seqs: int,
    context_len: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    kv_cache_dtype: str,
    dtype: torch.dtype,
    device: str = "cuda",):

    num_slots = (((context_len + 10) * num_seqs) + block_size - 1) // block_size * block_size
    num_blocks = num_slots // block_size

    context_lens = [(context_len + 1) for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

    flashattn_metadata = FlashAttentionBackend.make_metadata(
        is_prompt=False,
        use_cuda_graph = False,
        context_lens = context_lens,
        max_context_len = max_context_len,
        block_tables = block_tables,
        prompt_lens = None,
        prompt_lens_tensor = None,
        max_subquery_len = None,
        max_prompt_len = None,
        subquery_start_loc = None,
        seq_start_loc = None,
    )
    slot_mapping = random.sample(range(num_slots), context_len * num_seqs)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=device)
    attn_metadata = AttentionMetadata(
        0,
        0,
        num_seqs,
        None,
        flashattn_metadata,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
    )

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(num_blocks,
                                                            block_size,
                                                            1,
                                                            num_kv_heads,
                                                            head_size,
                                                            kv_cache_dtype,
                                                            dtype,
                                                            device=device)
    key_cache, value_cache = key_caches[0], value_caches[0]
    return attn_metadata, concat_kv_cache(key_cache, value_cache)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
        tp_size: int = 1,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide(output_size, tp_size)
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_method.create_weights(self,
                                          self.input_size,
                                          self.output_size_per_partition,
                                          self.input_size,
                                          self.output_size,
                                          self.params_dtype)
        # random initialize self weights
        self.weight.data.normal_(mean=0.0, std=0.5)
        
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.linear_method.apply_weights(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = output_parallel
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
        tp_size: int = 1,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        super().__init__(input_size, output_size, bias, False, skip_bias_add,
                         params_dtype, linear_method, tp_size)


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None,
                 tp_size: int = 8,
                 ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
            tp_size=tp_size,
        )
        self.o_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim // tp_size,
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
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


@torch.inference_mode()
def main(
    num_seqs: int,
    tp_size: int,
    context_len: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    hidden_size: int = 4096,
    device: str = "cuda",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    hidden_states = torch.randn((num_seqs, hidden_size),
                                dtype=dtype,
                                device=device)

    torch.set_default_dtype(dtype)

    mixtral_attn = MixtralAttention(
        hidden_size=hidden_size,
        num_heads=32,
        num_kv_heads=8,
        rope_theta=1e6,
        max_position=4096 * 32,
        tp_size = tp_size,
    )
    mixtral_attn = mixtral_attn.to(device=device, dtype=dtype)


    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        # attn_metadata, kv_cache = prepare_attn_metadata(
        #     num_seqs=num_seqs,
        #     context_len=context_len,
        #     block_size=16,
        #     num_kv_heads=8,
        #     head_size=128,
        #     kv_cache_dtype="auto",
        #     dtype=dtype,
        #     device=device,
        # )
        # positions = torch.randint(0, context_len, (num_seqs,),
        #                             dtype=torch.long, device=device)

        n_blocks, input_positions, attn_metadata = prepare_decode(
            num_seqs,
            context_len,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            kv_cache_dtype="auto",
            dtype=dtype,
            device=device,
        )

        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            65536,
            block_size=16,
            num_kv_heads = 8 // tp_size,
            head_size = 128)
        pin_memory = False
        kv_cache = torch.randn(kv_cache_shape,
                        dtype=dtype,
                        pin_memory=pin_memory,
                        device=device)

        for _ in range(num_iters):
            mixtral_attn(input_positions, hidden_states, kv_cache, attn_metadata)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the Mixtral attention layer.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)

    main(
        num_seqs=args.batch_size,
        tp_size=args.tp_size,
        context_len=args.context_len,
        dtype=STR_DTYPE_TO_TORCH_DTYPE["bfloat16"],
        seed=args.seed,
        do_profile=args.profile,
    )
