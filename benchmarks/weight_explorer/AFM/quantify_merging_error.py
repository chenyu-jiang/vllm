import argparse
import os
from typing import List

import numpy as np
import torch

import torch.nn.functional as F

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

def f_layer(w1, w2, w3, x):
    return (F.silu(x @ w1) * (x @ w3)) @ w2

def evaluate(merged_w1: torch.Tensor, merged_w2: torch.Tensor, merged_w3: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, x: List[torch.Tensor]):
    merged_out = f_layer(merged_w1, merged_w2, merged_w3, x)
    out = f_layer(w1, w2, w3, x)
    diff = torch.norm(merged_out - out, dim=1)
    return diff


def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
    return data

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
    w1_weight = exper_id_to_params[args.expert_id][f"w1"].to(args.device).t()
    w2_weight = exper_id_to_params[args.expert_id][f"w2"].to(args.device).t()
    w3_weight = exper_id_to_params[args.expert_id][f"w3"].to(args.device).t()
    # load merged weight
    w1_merged = torch.load(os.path.join(args.merged_weights_dir, f"w1_l{args.layer_id}", "merged_weight.pt")).to(args.device)
    w2_merged = torch.load(os.path.join(args.merged_weights_dir, f"w2_l{args.layer_id}", "merged_weight.pt")).to(args.device)
    w3_merged = torch.load(os.path.join(args.merged_weights_dir, f"w3_l{args.layer_id}", "merged_weight.pt")).to(args.device)
    # load data
    data = create_dataloaders(args)
    # train
    diff = evaluate(w1_merged, w2_merged, w3_merged, w1_weight, w2_weight, w3_weight, data)
    # save
    if save:
        save_path = os.path.join(args.output_dir, f"l{args.layer_id}_e{args.expert_id}.pt")
        torch.save(diff, save_path)
    # plot histogram
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({"Diff": diff.float().cpu().numpy()})
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.histplot(df, x="Diff", ax=ax)
    fig.savefig(os.path.join(args.output_dir, "histograms", f"l{args.layer_id}_e{args.expert_id}_histogram.pdf"), bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--merged_weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
