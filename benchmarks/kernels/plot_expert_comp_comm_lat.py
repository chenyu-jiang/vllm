import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob

# Read in data
expert_csvs = list(set(glob.glob('expert_computation*.csv')) - set(glob.glob('expert_computation*_cudagraph.csv')))
expert_csvs.sort()
print(expert_csvs)
expert_dfs = []
for path in expert_csvs:
    df = pd.read_csv(path)
    expert_dfs.append(df)

# Concatenate dataframes
expert_df = pd.concat(expert_dfs)
expert_df["batch_size"] = pd.Categorical(expert_df["batch_size"].astype(str),
                                        categories=[str(x) for x in sorted([int(x) for x in expert_df["batch_size"].unique()])],
                                        ordered=True)

# expert only plot
fig, ax = plt.subplots(figsize=(12, 4))
ax = sns.lineplot(x='batch_size', y='avg_latency', hue='tp_size', style="tp_size", data=expert_df, ax=ax)
ax.set_xticks([str(x) if idx % 2 == 0 else "" for idx, x in enumerate(sorted([int(x) for x in expert_df["batch_size"].unique()]))])
ax.set_xlabel('Batch Size')
ax.set_ylabel('Latency (ms)')
ax.set_title('Mixtral Expert Computation Latency (A100 80GB)')
# set legend title
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='TP Size')

plt.tight_layout()
plt.savefig('mixtral_expert_computation_latency_p4de.pdf')

# expert latency same number of GPUs
fig, ax = plt.subplots(figsize=(12, 4))
expert_df["scaled_latency"] = expert_df["avg_latency"] * expert_df["tp_size"]
ax = sns.lineplot(x='batch_size', y='scaled_latency', hue='tp_size', style="tp_size", data=expert_df, ax=ax)
ax.set_xticks([str(x) if idx % 2 == 0 else "" for idx, x in enumerate(sorted([int(x) for x in expert_df["batch_size"].unique()]))])
ax.set_xlabel('Batch Size')
ax.set_ylabel('Latency (ms)')
ax.set_title('Mixtral Expert Scaled Computation Latency (A100 80GB)')
# set legend title
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='TP Size')

plt.tight_layout()
plt.savefig('mixtral_expert_computation_latency_scaled_p4de.pdf')

expert_df["Type"] = "Expert TP" + expert_df["tp_size"].astype(str)
expert_df.drop(columns=["tp_size"], inplace=True)
expert_df.drop(columns=["scaled_latency"], inplace=True)

# Alltoall data
alltoall_csvs = list(set(glob.glob('alltoall_ws*.csv')))
alltoall_csvs.sort()
print(alltoall_csvs)
alltoall_dfs = []
for path in alltoall_csvs:
    df = pd.read_csv(path)
    alltoall_dfs.append(df)

# Concatenate dataframes
alltoall_df = pd.concat(alltoall_dfs)
alltoall_df["batch_size"] = pd.Categorical(alltoall_df["batch_size"].astype(str),
                                        categories=[str(x) for x in sorted([int(x) for x in alltoall_df["batch_size"].unique()])],
                                        ordered=True)
alltoall_df["Type"] = "AllToAll WorldSize" + alltoall_df["world_size"].astype(str)
alltoall_df.drop(columns=["world_size"], inplace=True)

# merge dataframes
all_df = pd.concat([expert_df, alltoall_df])

# append to expert plot
fig, ax = plt.subplots(figsize=(12, 4))
ax = sns.lineplot(x='batch_size', y='avg_latency', hue='Type', style="Type", data=all_df, ax=ax)
ax.set_xticks([str(x) if idx % 2 == 0 else "" for idx, x in enumerate(sorted([int(x) for x in all_df["batch_size"].unique()]))])
ax.tick_params(axis='x', which='both', labelsize=10)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Latency (ms)')
ax.set_title('Mixtral Latency (A100 80GB)')
plt.tight_layout()
plt.savefig('mixtral_expert_alltoall_latency_p4de.pdf')