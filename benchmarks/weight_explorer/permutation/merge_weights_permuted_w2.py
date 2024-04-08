import argparse
import os
from typing import List

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
        diff_mat = x @ (merged_weight - w.t())
        diff = torch.norm(diff_mat, dim=1)
        element_wise_diff = torch.mean(torch.abs(diff_mat))
        per_expert_diff_norm_mean.append(torch.mean(diff).item())
        per_expert_diff_norm_dist.append(list(diff.float().cpu().numpy()))
        per_expert_element_wise_diff.append(element_wise_diff.item())
    return per_expert_diff_norm_mean, per_expert_diff_norm_dist, per_expert_element_wise_diff

def get_output_subdir(args):
    return f"w{args.weight_id}_l{args.layer_id}" # + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")

def train(weights: List[torch.Tensor], xs: List[torch.Tensor], alpha=0.9):
    # compute inner products of xs
    inner_products = []
    for x in xs:
        x_cuda = x.cuda()
        inner_products.append((x_cuda.t() @ x_cuda).cpu())
    # reduce non-diagonal items
    for i in range(len(inner_products)):
        inner_i_cuda = inner_products[i].cuda()
        inner_products[i] = alpha * inner_i_cuda + (1 - alpha) * torch.diag_embed(torch.diagonal(inner_i_cuda))
    # compute the inverse of the sum of inner products
    inv_sum_inner_products = torch.inverse(sum(inner_products).float()).to(xs[0].dtype)
    # compute of sum of x^Tx w_i
    sum_xT_x_ws = []
    for x, w in zip(xs, weights):
        x_cuda = x.cuda()
        w_cuda = w.cuda()
        sum_xT_x_ws.append((x_cuda.t() @ x_cuda @ w_cuda.t()))
    sum_xT_x_w = sum(sum_xT_x_ws)
    # compute the optimal weight
    optimal_weight = inv_sum_inner_products @ sum_xT_x_w
    return optimal_weight


def create_dataloaders(args):
    data_per_expert = []
    for expert_id in args.experts_to_merge:
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=torch.bfloat16, device=args.device)
        data_per_expert.append(data)
    return data_per_expert

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
    w1_weights = []
    w3_weights = []
    weights = []
    for expert_id in args.experts_to_merge:
        if expert_id == 0:
            weights.append(exper_id_to_params[expert_id][f"w2"].to(args.device).float())
            w1_weights.append(exper_id_to_params[expert_id][f"w1"].to(args.device))
            w3_weights.append(exper_id_to_params[expert_id][f"w3"].to(args.device))
        else:
            w1, w2, w3 = torch.load(os.path.join(args.permuted_dir, f"l{args.layer_id}_e{expert_id}", "permuted_weights.pt"))
            weights.append(w2.to(args.device).float())
            w1_weights.append(w1.to(args.device).bfloat16())
            w3_weights.append(w3.to(args.device).bfloat16())
    # load data
    data = create_dataloaders(args)
    # project data using w1 and w3
    xs = []
    for (x, w1, w3) in zip(data, w1_weights, w3_weights):
        xw1 = x @ w1.t()
        xw1 = torch.nn.functional.silu(xw1, inplace=True)
        y = xw1.mul_(x @ w3.t())
        xs.append(y.float())
    del data
    del w1_weights
    del w3_weights
    # train
    weights = [w.cpu() for w in weights]
    xs = [x.cpu() for x in xs]
    merged_weight = train(weights, xs, alpha=args.alpha)
    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(merged_weight.bfloat16(), os.path.join(save_dir, "merged_weight.pt"))
    # evaluate
    weights = [w.cuda() for w in weights]
    xs = [x.cuda() for x in xs]
    per_expert_diff_mean, per_expert_diff_dist, per_expert_element_wise_diff = evaluate(merged_weight, weights, xs)
    for idx in range(len(args.experts_to_merge)):
        print(f"Expert {args.experts_to_merge[idx]} - Mean diff: {per_expert_diff_mean[idx]} - Element wise diff: {per_expert_element_wise_diff[idx]}")
    with open(os.path.join(save_dir, "diffs.txt"), "w") as f:
        for idx in range(len(args.experts_to_merge)):
            f.write(f"Expert {args.experts_to_merge[idx]} - Mean diff: {per_expert_diff_mean[idx]} - Element wise diff: {per_expert_element_wise_diff[idx]}\n")
    # plot histogram
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import seaborn as sns
    # experts = []
    # diffs = []
    # for expert_id in range(8):
    #     experts.extend([expert_id] * len(per_expert_diff_dist[expert_id]))
    #     diffs.extend(per_expert_diff_dist[expert_id])
    # df = pd.DataFrame({"Expert": experts, "Diff": diffs})
    # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # for i in range(8):
    #     sns.histplot(data=df[df["Expert"] == i], x="Diff", ax=axes[i // 4, i % 4])
    #     axes[i // 4, i % 4].set_title(f"Expert {i}")
    # fig.savefig(os.path.join(save_dir, "histogram.pdf"), bbox_inches="tight")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_id", type=int, default=2)
    parser.add_argument("--experts_to_merge", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--permuted_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.weight_id == 2 , "Weight ID must be 2"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    args.experts_to_merge = list(map(int, args.experts_to_merge.split(",")))

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
