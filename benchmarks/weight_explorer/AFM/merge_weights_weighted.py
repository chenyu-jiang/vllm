import argparse
import os
from typing import List

import pandas as pd

import wandb
import tqdm

import numpy as np
import torch

from vllm.model_executor.weight_utils import hf_model_weights_iterator

def _to_torch_dtype(dtype):
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype, None)
        if dtype is None:
            raise ValueError(f"Invalid dtype {dtype}")
    return dtype

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate(merged_weight: torch.Tensor, weights: List[torch.Tensor], xs: List[torch.Tensor]):
    per_expert_diff_norm_mean = []
    per_expert_diff_norm_dist = []
    per_expert_element_wise_diff = []
    for x, w in zip(xs, weights):
        diff_mat = x @ (merged_weight.t() - w.t())
        diff = torch.norm(diff_mat, dim=1)
        element_wise_diff = torch.mean(torch.abs(diff_mat))
        per_expert_diff_norm_mean.append(torch.mean(diff).item())
        per_expert_diff_norm_dist.append(list(diff.float().cpu().numpy()))
        per_expert_element_wise_diff.append(element_wise_diff.item())
    return per_expert_diff_norm_mean, per_expert_diff_norm_dist, per_expert_element_wise_diff

def get_output_subdir(args):
    return f"w{args.weight_id}_l{args.layer_id}" + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")


def calc_loss(target_model: torch.nn.Module, models: List[torch.nn.Module], xs: List[torch.Tensor], neuron_weights: List[torch.Tensor]):
    loss = 0
    for (x, model, neuron_weight) in zip(xs, models, neuron_weights):
        model_out = model(x)
        target_out = target_model(x)
        loss += torch.mean((model_out - target_out) ** 2, dim=0).dot(neuron_weight)
    return loss

def create_dataloaders(args):
    data_per_expert = []
    raw_data_per_expert = []
    for expert_id in range(8):
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        data_per_expert.append(data_loader)
        raw_data_per_expert.append(data)
    return data_per_expert, raw_data_per_expert

def load_neuron_weights_as_tensor(args):
    neuron_weights = []
    for expert_id in range(8):
        weights = pd.read_csv(os.path.join(args.neuron_weights_dir, f"l{args.layer_id}_e{expert_id}.csv"))["importance"]
        weights = torch.tensor(weights, dtype=_to_torch_dtype(args.dtype), device=args.device)
        neuron_weights.append(weights)
    return neuron_weights


def main_func(args, save=True):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                 fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    weights = []
    for expert_id in range(8):
        weights.append(exper_id_to_params[expert_id][f"w{args.weight_id}"].to(args.device))
    # create models
    models = []
    for expert_id in range(8):
        model = torch.nn.Linear(weights[0].shape[1], weights[0].shape[0], bias=False)
        model.weight.data.copy_(weights[expert_id])
        model = model.to(args.device).to(_to_torch_dtype(args.dtype))
        model.eval()
        models.append(model)
    # create target model
    target_model = torch.nn.Linear(weights[0].shape[1], weights[0].shape[0], bias=False)
    target_model = target_model.to(args.device).to(_to_torch_dtype(args.dtype))
    target_model.train()
    # load data
    dataloaders, raw_data = create_dataloaders(args)
    # load neuron weights
    neuron_weights = load_neuron_weights_as_tensor(args)
    # optimizer
    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)
    # train
    for epoch in tqdm.trange(args.num_epochs):
        for data in tqdm.tqdm(zip(*dataloaders), total=len(dataloaders[0])):
            optimizer.zero_grad()
            loss = calc_loss(target_model, models, list(data), neuron_weights)
            loss.backward()
            wandb.log({"loss": loss.item()})
            optimizer.step()
        lr_scheduler.step()

    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(target_model.state_dict(), os.path.join(save_dir, "target_model.pt"))
    # evaluate
    per_expert_diff_mean, per_expert_diff_dist, per_expert_element_wise_diff = evaluate(target_model.weight, weights, raw_data)
    for expert_id in range(8):
        print(f"Expert {expert_id} - Mean diff: {per_expert_diff_mean[expert_id]} - Element wise diff: {per_expert_element_wise_diff[expert_id]}")
    with open(os.path.join(save_dir, "diffs.txt"), "w") as f:
        for expert_id in range(8):
            f.write(f"Expert {expert_id} - Mean diff: {per_expert_diff_mean[expert_id]} - Element wise diff: {per_expert_element_wise_diff[expert_id]}\n")
    # plot histogram
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    experts = []
    diffs = []
    for expert_id in range(8):
        experts.extend([expert_id] * len(per_expert_diff_dist[expert_id]))
        diffs.extend(per_expert_diff_dist[expert_id])
    df = pd.DataFrame({"Expert": experts, "Diff": diffs})
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        sns.histplot(data=df[df["Expert"] == i], x="Diff", ax=axes[i // 4, i % 4])
        axes[i // 4, i % 4].set_title(f"Expert {i}")
    fig.savefig(os.path.join(save_dir, "histogram.pdf"), bbox_inches="tight")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--neuron_weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.weight_id == 1 or args.weight_id == 3, "Weight ID must be in {1, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    wandb.init(project="merging-weighted", dir=args.output_dir, config=args, name=get_output_subdir(args))
    main_func(args)
    wandb.finish()


if __name__ == "__main__":
    main()
