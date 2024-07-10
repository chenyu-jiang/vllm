import torch
import triton

from grouped_matmul_triton import grouped_matmul_kernel

from grouped_gemm.backend import gmm, get_ptrs

import glob

def read_problem_specs(path):
    prob_shapes = []
    with open(path, 'r') as f:
        for line in f:
            shape = line.strip().split(" ")[-1]
            m, n, k = shape.split("x")
            prob_shapes.append((int(m), int(n), int(k)))
    return prob_shapes

def prepare_torch_inputs(prob_shapes):
    group_m, group_n, group_k = zip(*prob_shapes)
    group_A = []
    group_B = []
    group_C = []
    group_size = len(group_m)
    batch_sizes = []
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device="cuda", dtype=torch.bfloat16)
        B = torch.rand((K, N), device="cuda", dtype=torch.bfloat16)
        C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        batch_sizes.append(M)
    # concat tensors
    group_A = torch.cat(group_A, dim=0).contiguous()
    group_B = torch.stack(group_B, dim=0).contiguous()
    group_C = torch.cat(group_C, dim=0).contiguous()
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long, device="cpu")
    return group_A, group_B, group_C, batch_sizes

def prepare_args_triton(A_concated, B_concated, C_concated, batch_sizes):
    device = torch.device('cuda')

    group_size = len(batch_sizes)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    curr_batch_size = 0
    for i in range(group_size):
        A = A_concated[curr_batch_size:curr_batch_size + batch_sizes[i].item()]
        B = B_concated[i]
        C = C_concated[curr_batch_size:curr_batch_size + batch_sizes[i].item()]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
        curr_batch_size += batch_sizes[i].item()

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable
    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size

def prepare_args_cublas(A_concated, B_concated, C_concated, batch_sizes):
    # concat tensors
    a_ptrs, b_ptrs, c_ptrs = get_ptrs(A_concated, B_concated, C_concated, batch_sizes, trans_b=False)
    a_ptrs = a_ptrs.to(A_concated.device)
    b_ptrs = b_ptrs.to(A_concated.device)
    c_ptrs = c_ptrs.to(A_concated.device)
    return a_ptrs, b_ptrs, c_ptrs, batch_sizes

# only launch the kernel, no tensor preparation here to remove all overhead
def triton_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
    )

def cublas_fn(a, b, c, a_ptrs, b_ptrs, c_ptrs, batch_sizes):
    gmm(a, b, batch_sizes,
        trans_a=False, trans_b=False,
        c=c, backend_type="cublas",
        a_ptrs=a_ptrs, b_ptrs=b_ptrs, c_ptrs=c_ptrs)

def ref_fn(a, b, batch_sizes):
    curr_batch_size = 0
    res = []
    for i in range(len(batch_sizes)):
        A = a[curr_batch_size:curr_batch_size + batch_sizes[i].item()]
        B = b[i]
        C = torch.matmul(A, B)
        res.append(C)
        curr_batch_size += batch_sizes[i].item()
    return torch.cat(res, dim=0)

def check_correctnss(spec_path):
    prob_shapes = read_problem_specs(spec_path)
    A_concated, B_concated, C_concated_tri, batch_sizes = prepare_torch_inputs(prob_shapes)
    
    # copy group C for cublas
    C_concated_cublas = torch.empty_like(C_concated_tri)

    triton_args = prepare_args_triton(A_concated, B_concated, C_concated_tri, batch_sizes)
    cublas_args = prepare_args_cublas(A_concated, B_concated, C_concated_cublas, batch_sizes)

    # first check correctness
    triton_fn(*triton_args)
    cublas_fn(A_concated, B_concated, C_concated_cublas, *cublas_args)
    ref_out = ref_fn(A_concated, B_concated, batch_sizes)

    assert torch.allclose(C_concated_tri, ref_out, atol=1e-2, rtol=0), f"triton output is incorrect"
    assert torch.allclose(C_concated_cublas, ref_out, atol=1e-2, rtol=0), f"cublas output is incorrect"


def benchmark(spec_path):
    prob_shapes = read_problem_specs(spec_path)
    A_concated, B_concated, C_concated, batch_sizes = prepare_torch_inputs(prob_shapes)

    triton_args = prepare_args_triton(A_concated, B_concated, C_concated, batch_sizes)
    cublas_args = prepare_args_cublas(A_concated, B_concated, C_concated, batch_sizes)

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        triton_results = triton.testing.do_bench_cudagraph(lambda: triton_fn(*triton_args))
        torch.cuda.synchronize()
        cublas_results = triton.testing.do_bench_cudagraph(lambda: cublas_fn(A_concated, B_concated, C_concated, *cublas_args))
    torch.cuda.synchronize()

    print("Triton results: mean {} us".format(triton_results * 1e3))
    print("Cublas results: mean {} us".format(cublas_results * 1e3))

if __name__ == "__main__":
    # for problem_spec in glob.glob("problem_specs/*.txt"):
    #     check_correctnss(problem_spec)
    # for problem_spec in glob.glob("problem_specs/*.txt"):
    for problem_spec in ["problem_specs/b384e1k1.txt"]:
        print(f"Running benchmark for {problem_spec}")
        benchmark(problem_spec)