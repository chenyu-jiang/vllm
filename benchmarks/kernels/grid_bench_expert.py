import subprocess

for tp_size in [2, 4, 8]:
    for use_cuda_graph in [False, True]:
        if tp_size == 1:
            args = ["python3", "benchmark_expert_computation.py",
                "--tp-size", str(tp_size)]
        else:
            args = ["torchrun", "--standalone", "--nnodes", "1",
                "--nproc_per_node", str(tp_size),
                "benchmark_expert_computation.py",
                "--tp-size", str(tp_size)]
        if use_cuda_graph:
            args.append("--use-cuda-graph")
        subprocess.run(args)