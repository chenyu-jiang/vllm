import os
import io
import subprocess

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ps = []
for batch_size in batch_sizes:
    p = subprocess.Popen(f"cd ~/vllm && python3 -m benchmarks.simulation.simulator -d /root/vllm/benchmarks/mixtral_expert_dataset -c /root/vllm/benchmarks/mixtral_cost_model -s FCFS --max-batch-size {batch_size}",
                     shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps.append(p)

if not os.path.exists("./simulation_results"):
    os.makedirs("./simulation_results")

fs = [open("./simulation_results/FCFS_bs{}_stdout.txt".format(bs), "w") for bs in batch_sizes]

readers = [io.TextIOWrapper(p.stdout, encoding='utf8') for p in ps]

while any([p.poll() is None for p in ps]):
    for bs, p, rd, f in zip(batch_sizes, ps, readers, fs):
        if p.poll() is None:
            # stream output to file
            char = rd.read(1)
            if char:
                f.write(char)
                f.flush()
        else:
            # check remaining output
            for char in iter(lambda: rd.read(1), ''):
                f.write(char)
                f.flush()

for p, f in zip(ps, fs):
    p.wait()
    f.close()