import argparse
import os
from typing import List

import numpy as np
import torch

from tqdm import tqdm

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from sklearn.linear_model import RANSACRegressor

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
    return f"w{args.weight_id}_l{args.layer_id}" + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")

def train(weights: List[torch.Tensor], xs: List[torch.Tensor], alpha=0.9):
    # compute inner products of xs
    inner_products = [x.t() @ x for x in xs]
    # reduce non-diagonal items
    for i in range(len(inner_products)):
        inner_products[i] = alpha * inner_products[i] + (1 - alpha) * torch.diag_embed(torch.diagonal(inner_products[i]))
    # compute the inverse of the sum of inner products
    inv_sum_inner_products = torch.inverse(sum(inner_products).float()).to(xs[0].dtype)
    # compute of sum of x^Tx w_i
    sum_xT_x_w = sum([x.t() @ x @ w.t() for x, w in zip(xs, weights)])
    # compute the optimal weight
    optimal_weight = inv_sum_inner_products @ sum_xT_x_w
    return optimal_weight

def train_ransac_scikit(weights: List[torch.Tensor], xs: List[torch.Tensor]):
    ys = [x @ w.t() for x, w in zip(xs, weights)]
    ys_cpu = [y.cpu().float().numpy() for y in ys]
    xs_cpu = [x.cpu().float().numpy() for x in xs]
    del ys
    xs_concat = np.concatenate(xs_cpu, axis=0)
    ys_concat = np.concatenate(ys_cpu, axis=0)
    # RANSAC
    ransac = RANSACRegressor(random_state=42)
    reg = ransac.fit(xs_concat, ys_concat)
    return torch.tensor(reg.estimator_.coef_, dtype=xs[0].dtype, device=xs[0].device)

def train_ransac(weights: List[torch.Tensor], xs: List[torch.Tensor],
                 max_iter=10000,
                 min_samples_multiplier=0.2,
                 residual_threshold_multiplier=0.1,
                 min_inliners_multiplier=0.5):
    total_samples = sum([x.shape[0] for x in xs])
    # first calculate the avg norm of original outputs
    avg_norms = [torch.mean(torch.linalg.norm(x @ w.t(), dim=1)).item() for (x, w) in zip(xs, weights)]
    iter_count = 0
    best_error = float("inf")
    best_weight = None
    with tqdm(total=max_iter) as pbar:
        while iter_count < max_iter:
            # random sample subset from each expert
            sampled_indices = [torch.tensor(np.random.choice(x.shape[0], int(x.shape[0] * min_samples_multiplier), replace=False)
                                            , dtype=torch.long, device=x.device) for x in xs]
            sampled_xs = [x[sampled_indices] for x, sampled_indices in zip(xs, sampled_indices)]
            # fit
            merged_weight, merged_bias = train(weights, sampled_xs)
            del sampled_indices
            del sampled_xs
            # eval
            confirmed_inliner_masks = []
            confirmed_inliner_errors = []
            confirmed_inliner_counts = []
            for (x, w, norm) in zip(xs, weights, avg_norms):
                diff = torch.linalg.norm(x @ (merged_weight - w.t()), dim=1)
                confirmed_inliner_masks.append(diff < norm * residual_threshold_multiplier)
                confirmed_inliner_errors.append(torch.sum(diff * confirmed_inliner_masks[-1]))
                confirmed_inliner_counts.append(torch.sum(confirmed_inliner_masks[-1]).item())
            n_inliners = sum(confirmed_inliner_counts)
            avg_error = (sum(confirmed_inliner_errors) / n_inliners).item()
            if n_inliners >= total_samples * min_inliners_multiplier:
                # found a acceptable model
                if avg_error < best_error:
                    best_error = avg_error
                    best_weight = merged_weight
                    tqdm.write(f"Found a better model with avg error {avg_error} and inliner ratio {n_inliners / total_samples}")
                else:
                    tqdm.write(f"Found a model with avg error {avg_error} and inliner ratio {n_inliners / total_samples}, but not better than previous best")
            else:
                tqdm.write(f"Failed to find a model with enough inliners: {n_inliners / total_samples}")
            iter_count += 1
            pbar.update(1)
    return best_weight

def create_dataloaders(args):
    data_per_expert = []
    for expert_id in range(8):
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
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
    weights = []
    for expert_id in range(8):
        weights.append(exper_id_to_params[expert_id][f"w{args.weight_id}"].to(args.device))
    # load data
    data = create_dataloaders(args)
    # train
    if args.regressor_type == "plain":
        merged_weight = train(weights, data)
    elif args.regressor_type == "ransac":
        merged_weight = train_ransac(weights, data)
    # save
    if save:
        save_dir = os.path.join(args.output_dir, args.regressor_type, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(merged_weight, os.path.join(save_dir, "merged_weight.pt"))
    # evaluate
    per_expert_diff_mean, per_expert_diff_dist, per_expert_element_wise_diff = evaluate(merged_weight, weights, data)
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
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--regressor_type", type=str, default="plain", choices=["plain", "ransac"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.weight_id == 1 or args.weight_id == 3, "Weight ID must be in {1, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
