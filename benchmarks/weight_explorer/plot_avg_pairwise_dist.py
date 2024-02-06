import torch

from vllm.model_executor.weight_utils import hf_model_weights_iterator

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

weight_dict = {}
for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1"):
    if "experts" in name:
        layer_id = int(name.split("layers.")[1].split(".")[0])
        expert_id = int(name.split(".")[5])
        if layer_id not in weight_dict:
            weight_dict[layer_id] = {}
        if expert_id not in weight_dict[layer_id]:
            weight_dict[layer_id][expert_id] = {}
        if "w1" in name:
            weight_dict[layer_id][expert_id]["w1"] = param
            assert "w2" not in name and "w3" not in name
        elif "w2" in name:
            weight_dict[layer_id][expert_id]["w2"] = param
            assert "w1" not in name and "w3" not in name
        elif "w3" in name:
            weight_dict[layer_id][expert_id]["w3"] = param
            assert "w1" not in name and "w2" not in name

w_list = []
layer_list = []
dist_list = []

for w_name in ["w1", "w2", "w3"]:
    # pair-wise distance
    for layer_id in range(32):
        for expert_id_a in range(8):
            for expert_id_b in range(8):
                if expert_id_a != expert_id_b:
                    wa = weight_dict[layer_id][expert_id_a][w_name].cuda()
                    wb = weight_dict[layer_id][expert_id_b][w_name].cuda()
                    dist = torch.norm(wa - wb).item()
                    w_list.append(w_name)
                    layer_list.append(layer_id)
                    dist_list.append(dist)

df = pd.DataFrame({"weight": w_list, "layer": layer_list, "dist": dist_list})

# plot avg pairwise distance distribution vs layer_id
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x="layer", y="dist", hue="weight", data=df, palette="Set3")
ax.set(xlabel="Layer ID", ylabel="Avg Pairwise Distance")

plt.savefig("avg_pairwise_dist.pdf", bbox_inches="tight")

w_list = []
layer_list = []
std_list = []

for w_name in ["w1", "w2", "w3"]:
    # pair-wise distance
    for layer_id in range(32):
        for expert_id_a in range(8):
            w = weight_dict[layer_id][expert_id_a][w_name].cuda()
            std = torch.std(w).item()
            w_list.append(w_name)
            layer_list.append(layer_id)
            std_list.append(std)

std_df = pd.DataFrame({"weight": w_list, "layer": layer_list, "std": std_list})
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x="layer", y="std", hue="weight", data=std_df, palette="Set3")
ax.set(xlabel="Layer ID", ylabel="Std Deviation")

plt.savefig("std_deviation.pdf", bbox_inches="tight")




