import subprocess

for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    subprocess.call(["python3", "benchmark_throughput.py", "--input-len", "512",
                      "--output-len", "512", "--model", "lmsys/vicuna-13b-v1.5-16k",
                      "--gpu-memory-utilization", str(frac)])