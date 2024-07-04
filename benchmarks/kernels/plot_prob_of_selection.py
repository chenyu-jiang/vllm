import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
batch_sizes = []
mean_activated_experts = []

res_map = {}
with open("./distribution_ne160_k6_solver_random.csv", "r") as f:
    first_line = f.readline()
    for line in f:
        act_exps, prob, bs, _ = line.split(",")
        act_exps = int(act_exps)
        prob = float(prob)
        bs = int(bs)
        if bs not in res_map:
            res_map[bs] = 0
        res_map[bs] += prob * act_exps

for bs in sorted(res_map.keys()):
    batch_sizes.append(bs)
    mean_activated_experts.append(res_map[bs])

df = pd.DataFrame({
    "Batch Size": batch_sizes,
    "Mean Activated Experts": mean_activated_experts
})

df["Batch Size"] = pd.Categorical(df["Batch Size"].astype(str), categories=[str(x) for x in sorted(df["Batch Size"].unique())], ordered=True)

# Plot the data
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax = sns.lineplot(data=df, x="Batch Size", y="Mean Activated Experts", marker="o", markersize=5, ax=ax)
ax.set_title("DeepSeekV2")

fig.savefig("prob_of_selection_DeepSeekV2.png", bbox_inches="tight", dpi=300)