from vllm import _custom_ops as ops
import vllm.envs as envs

from grouped_gemm.backend import gmm, get_ptrs
from vllm.model_executor.layers.fused_moe.fused_moe import *

def prepare_gmm_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    hs_tensors_buffer: torch.Tensor,
    intermediate_cache1: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    intermediate_cache3: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ret_rev_mapping: bool = False,
):
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens

    assert hidden_states.shape[0] == topk_ids.shape[0], "Batch size mismatch"
    # prepare input for each expert
    per_expert_inputs = {}
    if ret_rev_mapping:
        rev_mapping = {}
    for token_idx in range(topk_ids.shape[0]):
        for expert_idx in range(topk_ids.shape[1]):
            expert_id = topk_ids[token_idx, expert_idx].item()
            if expert_id not in per_expert_inputs:
                per_expert_inputs[expert_id] = []
            per_expert_inputs[expert_id].append(token_idx)
            if ret_rev_mapping:
                if expert_id not in rev_mapping:
                    rev_mapping[expert_id] = []
                rev_mapping[expert_id].append(token_idx * topk_ids.shape[1] + expert_idx)
    hs_tensors = []
    batch_sizes = []
    for expert_id in range(E):
        if expert_id not in per_expert_inputs:
            batch_sizes.append(0)
            continue
        token_indices = per_expert_inputs[expert_id]
        expert_input = hidden_states[token_indices]
        hs_tensors.append(expert_input)
        batch_sizes.append(len(token_indices))
    hs_tensors = torch.cat(hs_tensors, dim=0)
    hs_tensors_buffer.copy_(hs_tensors)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long, device="cpu")

    gmm1_ptrs = get_ptrs(hs_tensors_buffer, w1, intermediate_cache1, batch_sizes, trans_b=True)
    gmm2_ptrs = get_ptrs(intermediate_cache2, w2, intermediate_cache3, batch_sizes, trans_b=True)

    if ret_rev_mapping:
        return gmm1_ptrs, gmm2_ptrs, batch_sizes, rev_mapping
    return gmm1_ptrs, gmm2_ptrs, batch_sizes

def gmm_fused_experts_no_sum(
    hidden_states: torch.Tensor,
    hs_tensors: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    intermediate_cache1: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    intermediate_cache3: torch.Tensor,
    gmm1_ptrs: torch.Tensor,
    gmm2_ptrs: torch.Tensor,
    batch_sizes: torch.Tensor,
    backend_type="cublas"):
    # assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens

    a_ptrs, b_ptrs, c_ptrs = gmm1_ptrs
    gmm(hs_tensors, w1, batch_sizes,
        trans_a=False, trans_b=True if backend_type == "cublas" else False,
        c = intermediate_cache1,
        backend_type=backend_type,
        a_ptrs=a_ptrs,
        b_ptrs=b_ptrs,
        c_ptrs=c_ptrs)
    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
    a_ptrs, b_ptrs, c_ptrs = gmm2_ptrs
    gmm(intermediate_cache2, w2, batch_sizes,
        trans_a=False, trans_b=True if backend_type == "cublas" else False,
        c = intermediate_cache3,
        backend_type=backend_type,
        a_ptrs=a_ptrs,
        b_ptrs=b_ptrs,
        c_ptrs=c_ptrs)


def triton_fused_experts_no_sum(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  override_config: Optional[Dict[str, Any]] = None,
                  use_fp8: bool = False,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  out=None):
        # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        configs = get_moe_configs(E, w2.shape[2],
                                  "float8" if use_fp8 else None)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1.shape[2],
                                        topk_ids.shape[1],
                                        "float8" if use_fp8 else None)

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache1_full = intermediate_cache1
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2_full = intermediate_cache2
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3_full = intermediate_cache3

    compute_type = (tl.bfloat16
                    if hidden_states.dtype == torch.bfloat16 else tl.float16)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE:
            # will only happen in the last chunk
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], E))

        invoke_fused_moe_kernel(curr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1_scale,
                                w1_scale,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                topk_ids.shape[1],
                                config,
                                compute_type=compute_type,
                                use_fp8=use_fp8)

        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(intermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2_scale,
                                w2_scale,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8=use_fp8)

    if out is not None:
        out.copy_(intermediate_cache3_full)