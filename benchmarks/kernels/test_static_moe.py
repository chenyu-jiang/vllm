import torch
from vllm.model_executor.layers.fused_moe.fused_moe import *
from static_moe_utils import prepare_gmm_fused_experts, gmm_fused_experts_no_sum, triton_fused_experts_no_sum

from grouped_gemm.ops import gmm

def compare_with_triton(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
    num_shared_experts: int = 0,
    num_iters: int = 1,
    gmm_backend = "cublas",
    with_cuda_graph: bool = True,
) -> float:
    torch.manual_seed(0)
    init_dtype = torch.float16 if use_fp8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1 = torch.randn(num_experts + num_shared_experts,
                     shard_intermediate_size,
                     hidden_size,
                     dtype=init_dtype)
    assert w1.shape[0] == num_experts + num_shared_experts
    assert w1.shape[1] == shard_intermediate_size
    assert w1.shape[2] == hidden_size
    w2 = torch.randn(num_experts + num_shared_experts,
                     hidden_size,
                     shard_intermediate_size // 2,
                     dtype=init_dtype)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_fp8:
        w1_scale = torch.randn(num_experts + num_shared_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts + num_shared_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

    topk_weights_buffer = torch.empty(num_tokens, topk + num_shared_experts, dtype=dtype)
    topk_indices_buffer = torch.empty(num_tokens, topk + num_shared_experts, dtype=torch.int32)

    hs_tensors_buffer = torch.empty(num_tokens * (topk + num_shared_experts), hidden_size, dtype=dtype)
    intermediate_cache1_buffer = torch.empty(num_tokens * (topk + num_shared_experts), w1.shape[1], dtype=dtype)
    intermediate_cache2_buffer = torch.empty(num_tokens * (topk + num_shared_experts), w1.shape[1] // 2, dtype=dtype)
    intermediate_cache3_buffer = torch.empty(num_tokens * (topk + num_shared_experts), w2.shape[1], dtype=dtype)
    gmm1_ptr_a_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    gmm1_ptr_b_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    gmm1_ptr_c_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    gmm2_ptr_a_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    gmm2_ptr_b_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    gmm2_ptr_c_buffer = torch.empty(w1.shape[0], dtype=torch.uint64)
    batch_sizes_buffer = torch.empty(w1.shape[0], dtype=torch.long, device="cpu")

    res_t_buffer = torch.empty(num_tokens, (topk + num_shared_experts), w2.shape[1], dtype=dtype)

    rev_mapping = None

    if gmm_backend == "cutlass":
        w1t = w1.transpose(1, 2).contiguous()

    def prepare(i: int):
        nonlocal rev_mapping
        topk_weights, topk_ids = fused_topk(x, gating_output[i], topk, True)
        if num_shared_experts:
            topk_weights = torch.cat([topk_weights, torch.ones(num_shared_experts, dtype=dtype).expand(num_tokens, -1)], dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = torch.cat([topk_ids + num_shared_experts, torch.arange(0, num_shared_experts, 1).expand(num_tokens, -1)], dim=-1)
        topk_weights_buffer.copy_(topk_weights)
        topk_indices_buffer.copy_(topk_ids)

        gmm1_ptrs, gmm2_ptrs, batch_sizes, rev_mapping = prepare_gmm_fused_experts(
            x, w1, w2, hs_tensors_buffer, intermediate_cache1_buffer, intermediate_cache2_buffer, intermediate_cache3_buffer, topk_weights_buffer, topk_indices_buffer, ret_rev_mapping=True)
        gmm1_ptr_a_buffer.copy_(gmm1_ptrs[0])
        gmm1_ptr_b_buffer.copy_(gmm1_ptrs[1])
        gmm1_ptr_c_buffer.copy_(gmm1_ptrs[2])
        gmm2_ptr_a_buffer.copy_(gmm2_ptrs[0])
        gmm2_ptr_b_buffer.copy_(gmm2_ptrs[1])
        gmm2_ptr_c_buffer.copy_(gmm2_ptrs[2])
        batch_sizes_buffer.copy_(batch_sizes)

    def run():
        triton_fused_experts_no_sum(
            x,
            w1,
            w2,
            topk_weights_buffer,
            topk_indices_buffer,
            use_fp8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            out=res_t_buffer,
        )
        gmm_fused_experts_no_sum(
            x,
            hs_tensors_buffer,
            w1t if gmm_backend == "cutlass" else w1,
            w2,
            intermediate_cache1_buffer,
            intermediate_cache2_buffer,
            intermediate_cache3_buffer,
            (gmm1_ptr_a_buffer, gmm1_ptr_b_buffer, gmm1_ptr_c_buffer),
            (gmm2_ptr_a_buffer, gmm2_ptr_b_buffer, gmm2_ptr_c_buffer),
            batch_sizes_buffer,
            backend_type=gmm_backend,
        )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    if with_cuda_graph:
        # Capture with CUDA graph
        prepare(0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize()

    for i in range(num_iters):
        prepare(0)
        torch.cuda.synchronize()
        if with_cuda_graph:
            graph.replay()
        else:
            run()
        torch.cuda.synchronize()
        # rev map res_g_buffer
        res_g_remapped = torch.empty_like(intermediate_cache3_buffer)
        curr_bs = 0
        for expert_idx in range(batch_sizes_buffer.shape[0]):
            exp_tensor = intermediate_cache3_buffer[curr_bs:curr_bs + batch_sizes_buffer[expert_idx]]
            for j in range(exp_tensor.shape[0]):
                res_g_remapped[rev_mapping[expert_idx][j]] = exp_tensor[j]
            curr_bs += batch_sizes_buffer[expert_idx]
        allclose = torch.allclose(res_t_buffer.reshape(-1, res_t_buffer.shape[-1]), res_g_remapped, atol=1e-5)
        print(f"Iter {i}: {allclose}")
    if with_cuda_graph:
        graph.reset()

if __name__ == "__main__":
    with torch.device("cuda"):
        compare_with_triton(
            num_tokens=64,
            num_experts=8,
            shard_intermediate_size=2048,
            hidden_size=1024,
            topk=2,
            dtype=torch.bfloat16,
            use_fp8=False,
            num_shared_experts=0,
            num_iters=10,
            gmm_backend="cublas",
            with_cuda_graph=True,
        )