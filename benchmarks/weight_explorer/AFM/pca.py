# direct pca on expert weights

LAYER_ID = 0
WEIGHT_ID = 1

import torch
import numpy as np

from vllm.model_executor.weight_utils import hf_model_weights_iterator

exper_id_to_params = {}
for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                fall_back_to_pt=False):
    if f"layers.{LAYER_ID}" in name and "experts" in name:
        expert_id = int(name.split(".")[5])
        if expert_id not in exper_id_to_params:
            exper_id_to_params[expert_id] = {}
        for w_name in ["w1", "w2", "w3"]:
            if w_name in name:
                exper_id_to_params[expert_id][w_name] = param

weights_to_pca = []
for expert_id in range(8):
    weights_to_pca.append(exper_id_to_params[expert_id][f"w{WEIGHT_ID}"])

concated_weights = torch.cat(weights_to_pca, dim=0)
# svd
u, s, v = torch.svd(concated_weights.to(torch.float32).to("cuda"))
# calculate variance explained vs rank reduction
s_squared = s ** 2
var_explained = s_squared / s_squared.sum()
var_explained = var_explained.cpu().numpy()
var_explained_cum = np.cumsum(var_explained)


ranks = list(range(1, len(var_explained_cum) + 1))
types = ["All"] * len(ranks)
values = list(var_explained_cum)

# calculate svd for individual weights
for i, weight in enumerate(weights_to_pca):
    u, s, v = torch.svd(weight.to(torch.float32).to("cuda"))
    s_squared = s ** 2
    var_explained = s_squared / s_squared.sum()
    cum = np.cumsum(var_explained.cpu().numpy())
    ranks += list(range(1, len(cum) + 1))
    types += [f"Expert {i}"] * len(cum)
    values += list(cum)

import code
code.interact(local=locals())

# construct a dataframe
import pandas as pd

df = pd.DataFrame({"Rank": ranks, "Type": types, "Variance explained": values})

# plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Rank", y="Variance explained", hue="Type")

plt.xlabel("Reduced rank")
plt.ylabel("Variance explained")

plt.savefig("pca_variance_explained.pdf", format="pdf")

