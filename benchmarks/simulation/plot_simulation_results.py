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
    if schedule != "FCFS":
        act_experts = act_experts * 4
    if "mce" in fn:
        mce = int(os.path.basename(fn).split("_")[2][3:])
    else:
        mce = "N/A"
    data_list.append((schedule, bs, mce, act_experts, lat, throughput))

# process data_list to calculate throughput speedup
preprocessed_data_list = []
throughput_fcfs = {}
for schedule, bs, mce, act_experts, lat, throughput in data_list:
    if schedule == "FCFS":
        throughput_fcfs[bs] = throughput
for schedule, bs, mce, act_experts, lat, throughput in data_list:
    if schedule == "FCFS":
        speedup = 1.0
    else:
        speedup = throughput / throughput_fcfs[bs]
    preprocessed_data_list.append((schedule, bs, mce, act_experts, lat, throughput, speedup))

df = pd.DataFrame(preprocessed_data_list, columns=["schedule", "batch_size", "mce", "act_experts", "latency", "throughput", "throughput_speedup"])

df = df[(df["schedule"] == "PTL") | (df["schedule"] == "FCFS")]
df["batch_size"] = pd.Categorical(df["batch_size"].astype(str), categories=[str(x) for x in sorted(df["batch_size"].unique())], ordered=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sns.lineplot(data=df, x="batch_size", y="latency", hue="schedule", style="mce", ax=axes[0])
sns.lineplot(data=df, x="batch_size", y="throughput", hue="schedule", style="mce", ax=axes[1])
sns.lineplot(data=df, x="batch_size", y="throughput_speedup", hue="schedule", style="mce", ax=axes[2])
sns.lineplot(data=df, x="batch_size", y="act_experts", hue="schedule", style="mce", ax=axes[3])
axes[0].set_ylabel("Latency (ms)")
axes[1].set_ylabel("Throughput (tokens/s)")
axes[2].set_ylabel("Throughput Speedup")
axes[3].set_ylabel("Activated Experts")

for ax in axes:
    ax.set_xlabel("Batch Size")

# remove style legend
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[0:5], l[0:5], loc='best')

fig.tight_layout()
fig.savefig("simulation_results_ptl_only.pdf")

mce_df = df[df["schedule"] == "PTLW"]
mce_df = mce_df.copy()
mce_df["mce"] = pd.Categorical(mce_df["mce"].astype(int).astype(str), categories=[str(int(x)) for x in sorted(mce_df["mce"].unique())], ordered=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.lineplot(data=mce_df, x="mce", y="latency", hue="batch_size", ax=axes[0])
sns.lineplot(data=mce_df, x="mce", y="throughput", hue="batch_size", ax=axes[1])
sns.lineplot(data=mce_df, x="mce", y="act_experts", hue="batch_size", ax=axes[2])
axes[0].set_ylabel("Latency (ms)")
axes[1].set_ylabel("Throughput (tokens/s)")
axes[2].set_ylabel("Activated Experts")
axes[0].set_xlabel("Min Candidates per Expert")
axes[1].set_xlabel("Min Candidates per Expert")
axes[2].set_xlabel("Min Candidates per Expert")

fig.tight_layout()
fig.savefig("simulation_results_mce.pdf")

