import os
import io
import time
import subprocess

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
MCES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CMD = "cd ~/vllm && python3 -m benchmarks.simulation.simulator -d /root/vllm/benchmarks/simulation/expert_dataset -c /root/vllm/benchmarks/simulation/cost_model "

all_args = []
all_fns = []
for strategy in ["FCFS", "PTLW", "PT", "PTL"]:
    for bs in BATCH_SIZES:
        fname = "{}_bs{}".format(strategy, bs)
        args = ["-s", strategy, "--max-batch-size", str(bs), "--truncate-layers", "4"]
        if strategy == "PTLW":
            for mce in MCES:
                mce_args = args + ["--min-candidates-per-expert", str(mce)]
                mce_fname = fname + "_mce{}".format(mce)
                all_args.append(mce_args)
                all_fns.append(mce_fname)
        else:
            all_args.append(args)
            all_fns.append(fname)

# limit to 8 concurrent processes
ps = []
readers = []
fs = []
def _check_outputs():
    global ps, readers, fs
    for p, rd, f in zip(ps, readers, fs):
        if p.poll() is None:
            # stream output to file
            char = rd.read(1)
            if char:
                f.write(char)
                f.flush()
        else:
            # check remaining output and close
            for char in iter(lambda: rd.read(1), ''):
                f.write(char)
                f.flush()
            rd.close()
            f.close()
    ps = [p for p in ps if p.poll() is None]
    readers = [rd for rd in readers if not rd.closed]
    fs = [f for f in fs if not f.closed]
    _print_current_running_fns()
    time.sleep(0.1)

_last_printed = ""
def _print_current_running_fns():
    global fs
    global _last_printed
    s = "Running: {}".format(", ".join([os.path.basename(f.name) for f in fs]))
    if s != _last_printed:
        _last_printed = s
        os.system("clear")
        print(s, end="", flush=True)

if not os.path.exists("./simulation_results"):
    os.makedirs("./simulation_results")
for args, fn in zip(all_args, all_fns):
    # check if we already have results for this
    if os.path.exists("./simulation_results/{}_stdout.txt".format(fn)):
        skip = False
        with open("./simulation_results/{}_stdout.txt".format(fn), "r") as f:
            for line in f:
                if "Avg throughput:" in line:
                    skip = True
                    break
        if skip:
            continue
    while len(ps) >= 8:
        _check_outputs()
    p = subprocess.Popen(CMD + " ".join(args),
                     shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps.append(p)
    readers.append(io.TextIOWrapper(p.stdout, encoding='utf8'))
    fs.append(open("./simulation_results/{}_stdout.txt".format(fn), "w"))

while len(ps) > 0:
    _check_outputs()