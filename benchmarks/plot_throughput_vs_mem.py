import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./throughput_vs_memory.csv")

A100_MEM_SIZE = 40960

df["Tokens Per Second"] = df["throughput_tokens"]
df["Available GPU Memory"] = df["memory_frac"] * A100_MEM_SIZE
df["Available GPU Memory"] = df["Available GPU Memory"].astype(int)
df["Available GPU Memory"] = pd.Categorical(df["Available GPU Memory"].astype(str), 
                                    categories=[str(x) for x in sorted([int(x) for x in df["Available GPU Memory"].unique()])],
                                    ordered=True)
df["Model"] = df["model"]

plt.clf()
fig, ax = plt.subplots(figsize=(6, 2))
sns.lineplot(x="Available GPU Memory", y="Tokens Per Second", style="Model", data=df, ax=ax, errorbar=None, markers="o")
# ax.set_xscale('log', base=2)
sns.move_legend(ax, loc="upper left", labelspacing=0.2, columnspacing=0.2)
ax.get_legend().set_title(None)

plt.setp(ax.get_legend().get_texts(), fontsize='7')
plt.setp(ax.get_legend().get_title(), fontsize='7')
# plt.tight_layout()
plt.savefig('./throughput_vs_mem.pdf', bbox_inches='tight')