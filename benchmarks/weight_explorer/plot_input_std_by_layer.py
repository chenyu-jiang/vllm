import os
import numpy as np

import tqdm

import torch


def calc_input_std(data_dir, expert_id, layer_id):
    data = np.load(os.path.join(data_dir, f"expert_activations_e{expert_id}_l{layer_id}_0.npz"))["arr_0"]
    data = torch.tensor(data.reshape(-1, 4096)).cuda().to(torch.bfloat16)
    # calculate std for each feature
    data = torch.std(data, dim=1)
    norm = torch.norm(data)
    return norm


def main():
    data_dir = "/root/expert_data"
    with open("input_std_by_layer.csv", "w") as f:
        f.write("Expert,Layer,Norm\n")
        for expert_id in tqdm.trange(8, desc="Expert"):
            for layer_id in tqdm.trange(32, desc="Layer"):
                norm = calc_input_std(data_dir, expert_id, layer_id)
                f.write(f"{expert_id},{layer_id},{norm}\n")

def main_plot():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv("input_std_by_layer.csv")
    # average over experts
    # df = df.groupby(["Layer"]).mean().reset_index()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(x="Layer", y="Norm", hue="Expert", data=df)
    ax.set(xlabel="Layer ID", ylabel="Input Std")
    plt.savefig("input_std_by_layer.pdf", bbox_inches="tight")

# main()
main_plot()

