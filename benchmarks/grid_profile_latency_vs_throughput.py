
import subprocess

# for num_seqs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
# for num_seqs in [32, 64, 128, 256, 384, 512]:
for num_seqs in [1, 2, 4, 8, 16]:
    if num_seqs <= 16:
        n_samples = 50
    elif num_seqs < 32:
        n_samples = 200
    else:
        n_samples = 1000
    with open(f"./mixtral_benchmark/throughput_inlen128_outlen512_ns{num_seqs}.txt", "w") as f:
        p = subprocess.Popen(" ".join(["HF_HOME=/nfs/hf_cache",
            "python3", "benchmark_throughput.py",
            "--input-len", "128",
            "--output-len", "512",
            "--model", "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "-tp", "4",
            "--num-prompts", str(n_samples),
            "--max-num-seqs", str(num_seqs),
        ]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True)
        while p.poll() is None:
            # stream output to file
            for line in iter(p.stdout.readline, b''):
                print(line.decode('utf-8'))
                f.write(line.decode('utf-8'))
                f.flush()
        # check remaining output
        for line in iter(p.stdout.readline, b''):
            print(line.decode('utf-8'))
            f.write(line.decode('utf-8'))
            f.flush()