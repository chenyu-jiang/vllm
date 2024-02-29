import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
corr_df = pd.read_csv("./error_predictors/spearman_corr.csv")

# Plot the data
sns.set_theme(style="whitegrid")
ax = sns.lineplot(x="layer_id", y="corr", hue="expert_id", data=corr_df, marker="o")
ax.set_title("Spearman Correlation of Error Predictors")

ax.set_xlabel("Layer ID")
ax.set_ylabel("Spearman Correlation")

plt.savefig("spearman_corr.pdf", bbox_inches="tight")