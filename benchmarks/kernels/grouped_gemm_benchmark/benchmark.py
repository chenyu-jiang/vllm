import os
import torch
import triton
import tqdm

from grouped_matmul_triton import grouped_matmul_kernel

from grouped_gemm.backend import gmm, get_ptrs

import subprocess

def get_prob_spec(m, n, k):
    assert len(k) == 2
    return f"0 {m}x{n}x{k[0]}\n1 {m}x{n}x{k[1]}\n2 {m}x{n}x{k[0]}\n3 {m}x{n}x{k[1]}".strip()

def read_problem_specs(path):
    prob_shapes = []
    with open(path, 'r') as f:
        for line in f:
            shape = line.strip().split(" ")[-1]
            m, n, k = shape.split("x")
            prob_shapes.append((int(m), int(n), int(k)))
    return prob_shapes

def prepare_torch_inputs_diff_k(prob_shapes):
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
    return group_A, group_B, group_C, batch_sizes

def prepare_torch_inputs(prob_shapes):
    group_A, group_B, group_C, batch_sizes = prepare_torch_inputs_diff_k(prob_shapes)
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

def prepare_args_triton_diffk(group_A, group_B, group_C):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B) == len(group_C)

    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        C = group_C[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
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

def is_diff_k(prob_shapes):
    group_k = prob_shapes[0][-1]
    for shape in prob_shapes:
        if shape[-1] != group_k:
            return True
    return False

def benchmark(spec_path, silent=False):
    prob_shapes = read_problem_specs(spec_path)
    if is_diff_k(prob_shapes):
        group_A, group_B, group_C, _ = prepare_torch_inputs_diff_k(prob_shapes)
        triton_args = prepare_args_triton_diffk(group_A, group_B, group_C)
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            triton_results = triton.testing.do_bench_cudagraph(lambda: triton_fn(*triton_args))
        torch.cuda.synchronize()
        if not silent:
            print("Triton results: mean {} us".format(triton_results * 1e3))
        cublas_results = None
    else:
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

        if not silent:
            print("Triton results: mean {} us".format(triton_results * 1e3))
            print("Cublas results: mean {} us".format(cublas_results * 1e3))
    return triton_results, cublas_results

def benchmark_cutlass(spec_path, sort=False, capture_output=False):
    if sort:
        p = subprocess.run(f"/root/cutlass/build/examples/24_gemm_grouped/24_gemm_grouped --benchmark={os.path.abspath(spec_path)} --verbose --iterations=10000 --sort-problems=true", shell=True, capture_output=capture_output)
    else:
        p = subprocess.run(f"/root/cutlass/build/examples/24_gemm_grouped/24_gemm_grouped --benchmark={os.path.abspath(spec_path)} --verbose --iterations=10000", shell=True, capture_output=capture_output)
    if capture_output:
        output = p.stdout.decode("utf-8")
        for line in output.split("\n"):
            if "Batched Runtime:" in line:
                res_batched = float(line.strip().split(":")[1].strip().split(" ")[0].strip())
            if "Grouped Runtime:" in line:
                res_grouped = float(line.strip().split(":")[1].strip().split(" ")[0].strip())
            if "total threadblock tiles" in line:
                total_tiles = int(line.strip().split(" ")[0].strip())
            if "Threadblock Tile Shape" in line:
                shape_mnk = line.strip().split(":")[1].strip()
        return res_batched, res_grouped, total_tiles, shape_mnk

if __name__ == "__main__":
    # for problem_spec in glob.glob("problem_specs/*.txt"):
    #     check_correctnss(problem_spec)
    # for problem_spec in glob.glob("problem_specs/*.txt"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_spec", type=str, required=False, default="problem_specs/b384e1k1.txt")
    parser.add_argument("--sort", action="store_true", help="sort the problem specs by k size")
    parser.add_argument("--m-range", type=str, required=False, help="range of m values")
    parser.add_argument("--m-step", type=int, default=128, help="step size for m")
    parser.add_argument("--n", type=int, default=1152, help="n value")
    parser.add_argument("--k", type=int, nargs=2, default=[128, 1024], help="k values")
    parser.add_argument("--output", type=str, default="./results.csv", help="output file for benchmark results")
    args = parser.parse_args()
    if args.m_range:
        for m in (pbar := tqdm.trange(*map(int, args.m_range.split(",")), args.m_step)):
            pbar.set_description(f"m={m}")
            prob_spec = "./problem_specs_autogen/tmp.txt"
            if not os.path.exists(args.output):
                with open(args.output, "w") as f:
                    f.write("m,n,k0,k1,k2,label,balanced,time,cutlass_sorted,cutlass_total_tiles,cutlass_tile_shape\n")
            k2 = (args.k[0] + args.k[1]) // 2
            with open(args.output, "a") as fout:
                # imbalanced case
                prob_spec_str_imbalanced = get_prob_spec(m, args.n, args.k)
                with open("./problem_specs_autogen/tmp.txt", "w") as f:
                    f.write(prob_spec_str_imbalanced)
                triton_time, cublas_time = benchmark(prob_spec, silent=True)
                fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},Triton,False,{triton_time},-,-,-\n")
                fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},cuBLAS,False,{cublas_time},-,-,-\n")
                for sort in [False, True]:
                    res_batched, res_grouped, total_tiles, shape_mnk = benchmark_cutlass(prob_spec, sort, capture_output=True)
                    fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},CUTLASS_BATCH,False,{res_batched},{sort},{total_tiles},{shape_mnk}\n")
                    fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},CUTLASS_GROUP,False,{res_grouped},{sort},{total_tiles},{shape_mnk}\n")
                # balanced case
                prob_spec_str_balanced = get_prob_spec(m, args.n, [k2, k2])
                with open("./problem_specs_autogen/tmp.txt", "w") as f:
                    f.write(prob_spec_str_balanced)
                triton_time, cublas_time = benchmark(prob_spec, silent=True)
                fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},Triton,True,{triton_time},-,-,-\n")
                fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},cuBLAS,True,{cublas_time},-,-,-\n")
                for sort in [False, True]:
                    res_batched, res_grouped, total_tiles, shape_mnk = benchmark_cutlass(prob_spec, sort, capture_output=True)
                    fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},CUTLASS_BATCH,True,{res_batched},{sort},{total_tiles},{shape_mnk}\n")
                    fout.write(f"{m},{args.n},{args.k[0]},{args.k[1]},{k2},CUTLASS_GROUP,True,{res_grouped},{sort},{total_tiles},{shape_mnk}\n")
    else:
        for problem_spec in [args.problem_spec]:
            print(f"Running benchmark for {problem_spec}")
            benchmark(problem_spec)
            benchmark_cutlass(problem_spec, args.sort)