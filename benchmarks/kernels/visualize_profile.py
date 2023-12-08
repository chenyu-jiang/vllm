import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./paged_attention_profile.csv")

# encoder_fw_df = df[df['Stage'] == 'Encoder FW'].drop("Stage", axis=1)
df["Sequence Per Second"] = df["throughput_seqs"]
df["KV Size (GB) Per Second"] = df["throughput_kv_MB"] / 1000.0
df["Context Length"] = df["context_len"].astype(str)
print(sorted([int(x) for x in df["batch_size"].unique()]))

df["Batch Size"] = pd.Categorical(df["batch_size"].astype(str), 
                                    categories=[str(x) for x in sorted([int(x) for x in df["batch_size"].unique()])],
                                    ordered=True)


# import code
# code.interact(local=locals())


# norm = plt.Normalize(df["Context Length"].min(), df["Context Length"].max())
# sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
# sm.set_array([])

plt.clf()
fig, ax = plt.subplots(figsize=(6, 2))
# plt.title("Encoder FW")
sns.lineplot(x="Batch Size", y="Sequence Per Second", hue="Context Length", style="Context Length", palette="viridis", dashes=False, data=df, ax=ax, errorbar=None, markers="o")
# ax.set_xscale('log', base=2)
sns.move_legend(ax, loc="lower left", bbox_to_anchor=[1, 0] ,ncol=2, labelspacing=0.2, columnspacing=0.2)
plt.setp(ax.get_legend().get_texts(), fontsize='9')
plt.setp(ax.get_legend().get_title(), fontsize='9')
# plt.tight_layout()
plt.savefig('./pagedattn_throughput_seq.png', dpi=300, bbox_inches='tight')
ax.set_yscale('log', base=2)
plt.savefig('./pagedattn_throughput_seq_log.png', dpi=300, bbox_inches='tight')


A100_MEM_BW = 1555.0
A100_PCIE_BW = 64.0
A100_PCIE_BW_PROFILED = 26.236258
A100_NVLINK_BW = 600.0
plt.clf()
fig, ax = plt.subplots(figsize=(6, 2))
# plt.title("Encoder FW")
sns.lineplot(x="Batch Size", y="KV Size (GB) Per Second", hue="Context Length", style="Context Length", palette="viridis", dashes=False, data=df, ax=ax, errorbar=None, markers="o")
# ax.set_xscale('log', base=2)
# ax.axhline(y=A100_MEM_BW, color='black', linestyle='--', linewidth=1)
# ax.text(0, A100_MEM_BW - 10, 'A100 Memory Bandwidth', ha='left', va='top', color='black', fontsize=9)
ax.axhline(y=A100_PCIE_BW, color='black', linestyle='--', linewidth=1)
ax.text(0, A100_PCIE_BW + 10, 'A100 PCIe Bandwidth', ha='left', va='bottom', color='black', fontsize=9)
ax.axhline(y=A100_NVLINK_BW, color='black', linestyle='--', linewidth=1)
ax.text(0, A100_NVLINK_BW - 10, 'A100 NVLink Bandwidth', ha='left', va='top', color='black', fontsize=9)
# ax.axhline(y=A100_PCIE_BW_PROFILED, color='black', linestyle='--', linewidth=1)
# ax.text(0, A100_PCIE_BW_PROFILED + 10, 'A100 PCIe Bandwidth (Profiled)', ha='left', va='bottom', color='black', fontsize=9)
sns.move_legend(ax, loc="lower left", bbox_to_anchor=[1, 0] ,ncol=2, labelspacing=0.2, columnspacing=0.2)
plt.setp(ax.get_legend().get_texts(), fontsize='9')
plt.setp(ax.get_legend().get_title(), fontsize='9')
# plt.tight_layout()
plt.savefig('./pagedattn_throughput_kv.png', dpi=300, bbox_inches='tight')
ax.set_yscale('log', base=2)
plt.savefig('./pagedattn_throughput_kv_log.png', dpi=300, bbox_inches='tight')