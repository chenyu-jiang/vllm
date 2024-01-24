import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

PREFIX = "/root/vllm/benchmarks/simulation/simulation_results"

fns = glob.glob(f"{PREFIX}/*.txt")
fns.sort()

data_list = []
for fn in fns:
    act_experts = None
    lat = None
    throughput = None
    with open(fn, "r") as f:
        for line in f:
            if "Avg activated experts per batch:" in line:
                act_experts = float(line.strip().split()[-1])
            elif "Avg latency:" in line:
                lat = float(line.strip().split()[-2])
            elif "Avg throughput:" in line:
                throughput = float(line.strip().split()[-2])
    if act_experts is None or lat is None or throughput is None:
        continue
    schedule = os.path.basename(fn).split("_")[0]
    bs = int(os.path.basename(fn).split("_")[1][2:])
    if "mce" in fn:
        act_experts = act_experts * 4
        mce = int(os.path.basename(fn).split("_")[2][3:])
        # if mce != 128:
        #     continue
    else:
        mce = None
    data_list.append((schedule, bs, mce, act_experts, lat, throughput))

df = pd.DataFrame(data_list, columns=["schedule", "batch_size", "mce", "act_experts", "latency", "throughput"])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.lineplot(data=df, x="batch_size", y="latency", hue="schedule", ax=axes[0])
sns.lineplot(data=df, x="batch_size", y="throughput", hue="schedule", ax=axes[1])
sns.lineplot(data=df, x="batch_size", y="act_experts", hue="schedule", ax=axes[2])
axes[0].set_ylabel("Latency (ms)")
axes[1].set_ylabel("Throughput (tokens/s)")
axes[2].set_ylabel("Activated Experts")
axes[0].set_xlabel("Batch Size")
axes[1].set_xlabel("Batch Size")
axes[2].set_xlabel("Batch Size")

fig.tight_layout()
fig.savefig("simulation_results.pdf")

mce_df = df[(df["batch_size"] == 32) & (df["schedule"] == "PTLW")]
mce_df = mce_df.copy()
mce_df["mce"] = pd.Categorical(mce_df["mce"].astype(int).astype(str), categories=[str(int(x)) for x in sorted(mce_df["mce"].unique())], ordered=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.lineplot(data=mce_df, x="mce", y="latency", ax=axes[0])
sns.lineplot(data=mce_df, x="mce", y="throughput", ax=axes[1])
sns.lineplot(data=mce_df, x="mce", y="act_experts", ax=axes[2])
axes[0].set_ylabel("Latency (ms)")
axes[1].set_ylabel("Throughput (tokens/s)")
axes[2].set_ylabel("Activated Experts")
axes[0].set_xlabel("Min Candidates per Expert")
axes[1].set_xlabel("Min Candidates per Expert")
axes[2].set_xlabel("Min Candidates per Expert")

fig.tight_layout()
fig.savefig("simulation_results_mce.pdf")

