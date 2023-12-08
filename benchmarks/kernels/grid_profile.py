import subprocess

for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for context_len in [2 ** i for i in range(8, 21)]:
        subprocess.run([
            "python", "benchmark_paged_attention.py",
            "--batch-size", str(batch_size),
            "--context-len", str(context_len),
        ])