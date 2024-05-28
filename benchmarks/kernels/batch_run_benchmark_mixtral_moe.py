import subprocess
import time

profile=False

cmds = []
for num_experts in [1, 2, 4, 6, 8]:
    for tp_size in [1, 2, 4, 8]:
        cmd = f"python3 benchmark_mixtral_moe.py --num_experts {num_experts} --tp_size {tp_size}" + (" --profile" if profile else "")
        cmds.append(cmd)

num_gpus = 8
idle_gpus = set(range(num_gpus))
processes = [None for _ in range(num_gpus)]
for cmd in cmds:
    while len(idle_gpus) == 0:
        # wait for a process to finish
        for i in range(len(processes)):
            assert processes[i] is not None
            if processes[i].poll() is not None:
                idle_gpus.add(i)
                processes[i] = None
                break
        if len(idle_gpus) == 0:
            # still no idle gpu, wait
            time.sleep(1)
    idle_gpu = idle_gpus.pop()
    # launch
    gpu_cmd = "CUDA_VISIBLE_DEVICES=" + str(idle_gpu) + " " + cmd
    p = subprocess.Popen(gpu_cmd, shell=True)
    processes[idle_gpu] = p
