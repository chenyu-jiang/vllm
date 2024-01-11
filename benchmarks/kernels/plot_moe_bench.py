import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

benchmark_csvs = glob.glob("./mixtral_moe_*.csv")
dfs = []
for csv in benchmark_csvs:
    df = pd.read_csv(csv)
    dfs.append(df)

df = pd.concat(dfs)

df["throughput"] = (df["batch_size"] * (8 / (df["dp_size"] * df["tp_size"]))) / (df["avg_latency"] / 1000)

df["Batch Size"] = pd.Categorical(df["batch_size"].astype(str), categories=[str(x) for x in sorted(df["batch_size"].unique())], ordered=True)

df["Latency (ms)"] = df["avg_latency"]

df["Parallel Method"] = df["parallel_method"]

df["Parallelism"] = df.apply(lambda x: f"DPx{int(8 / x['tp_size'])};TPx{x['tp_size']}", axis=1)
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x="Batch Size", y="throughput", hue="Parallelism", style="Parallel Method", data=df, ax=ax)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Single MoE Layer Throughput (tokens/s)")
ax.set_xticks([str(x) if idx % 2 == 0 else "" for idx, x in enumerate(sorted([int(x) for x in df["Batch Size"].unique()]))])
ax.tick_params(axis='x', which='both', labelsize=10)
plt.tight_layout()
plt.savefig("moe_benchmark_throughput.pdf")

# df["Parallelism"] = df.apply(lambda x: f"DPx{x['dp_size']};TPx{x['tp_size']}", axis=1)
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x="Batch Size", y="avg_latency", hue="Parallelism", style="Parallel Method", data=df, ax=ax)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Single MoE Layer Latency (ms)")
ax.set_xticks([str(x) if idx % 2 == 0 else "" for idx, x in enumerate(sorted([int(x) for x in df["Batch Size"].unique()]))])
ax.tick_params(axis='x', which='both', labelsize=10)
plt.tight_layout()
plt.savefig("moe_benchmark_latency.pdf")
