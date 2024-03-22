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


def evaluate(
    merged_weight: torch.Tensor,
    weights: List[torch.Tensor],
    xs: List[torch.Tensor],
    threshold=1e-4,
):
    per_expert_diff_norm_mean = []
    per_expert_element_wise_diff = []
    per_expert_inliners = []
    for x, w in zip(xs, weights):
        diff_mat = x @ (merged_weight - w.t())
        n_inliners = torch.sum(torch.abs(diff_mat) < threshold).item()
        diff_norm = torch.norm(diff_mat, dim=1)
        element_wise_diff = torch.mean(torch.abs(diff_mat))
        per_expert_diff_norm_mean.append(torch.mean(diff_norm).item())
        per_expert_element_wise_diff.append(element_wise_diff.item())
        per_expert_inliners.append(n_inliners.item())
    return (
        per_expert_diff_norm_mean,
        per_expert_element_wise_diff,
        per_expert_inliners,
    )


def get_output_subdir(args):
    return f"w{args.weight_id}_l{args.layer_id}"


def fit_once(weights: List[torch.Tensor], xs: List[torch.Tensor], alpha=1.0):
    # compute inner products of xs
    inner_products = [x.t() @ x for x in xs]
    # reduce non-diagonal items
    for i in range(len(inner_products)):
        inner_products[i] = alpha * inner_products[i] + (
            1 - alpha
        ) * torch.diag_embed(torch.diagonal(inner_products[i]))
    # compute the inverse of the sum of inner products
    inv_sum_inner_products = torch.inverse(sum(inner_products).float()).to(
        xs[0].dtype
    )
    # compute of sum of x^Tx w_i
    sum_xT_x_w = sum([x.t() @ x @ w.t() for x, w in zip(xs, weights)])
    # compute the optimal weight
    optimal_weight = inv_sum_inner_products @ sum_xT_x_w
    return optimal_weight


def ransac(
    weights: List[torch.Tensor],
    xs: List[torch.Tensor],
    sample_ratio=0.125,
    max_iterations=1000,
    inliner_threshold=1e-4,
    alpha=1.0,
):
    # get total number of samples
    total_samples = sum([x.shape[0] for x in xs])
    n_samples = int(total_samples * sample_ratio)
    cumsum_samples = np.cumsum([x.shape[0] for x in xs])

    best_inliners = 0
    best_weight = None
    for _ in range(max_iterations):
        # random sample
        sample_indices = sorted(
            np.random.choice(total_samples, n_samples, replace=False).tolist()
        )
        # construct sampled xs
        sampled_xs = []
        per_expert_indices = [[] for _ in range(len(xs))]
        for index in sample_indices:
            expert_id = np.argmax(cumsum_samples > index)
            per_expert_indices[expert_id].append(
                index - cumsum_samples[expert_id - 1]
                if expert_id > 0
                else index
            )
        for expert_id, indices in enumerate(per_expert_indices):
            sampled_xs.append(xs[expert_id][indices])
        # fit
        merged_weight = fit_once(weights, sampled_xs, alpha=alpha)
        # evaluate
        _, _, per_expert_inliners = evaluate(
            merged_weight, weights, xs, threshold=inliner_threshold
        )
        total_inliners = sum(per_expert_inliners)
        if total_inliners > best_inliners:
            best_inliners = total_inliners
            best_weight = merged_weight
    return best_weight


def create_dataloaders(args):
    data_per_expert = []
    for expert_id in args.experts_to_merge:
        data = np.load(
            os.path.join(
                args.data_dir,
                f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz",
            )
        )["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(
            data, dtype=_to_torch_dtype(args.dtype), device=args.device
        )
        data_per_expert.append(data)
    return data_per_expert


def main_func(args, save=True):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator(
        "mistralai/Mixtral-8x7B-v0.1", fall_back_to_pt=False
    ):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    weights = []
    for expert_id in args.experts_to_merge:
        weights.append(
            exper_id_to_params[expert_id][f"w{args.weight_id}"]
            .to(args.device)
            .float()
        )
    # load data
    data = create_dataloaders(args)
    # train
    best_weight = ransac(
        weights,
        data,
        sample_ratio=args.sample_ratio,
        max_iterations=args.max_iterations,
        inliner_threshold=args.inliner_threshold,
        alpha=args.alpha,
    )
    # evaluate
    (
        per_expert_diff_mean,
        per_expert_element_wise_diff,
        per_expert_inliners,
    ) = evaluate(best_weight, weights, data)
    print("==" * 20)
    print("Per expert inliners: {}".format(per_expert_inliners))
    for idx in range(len(args.experts_to_merge)):
        print(
            f"Expert {args.experts_to_merge[idx]} - Mean diff: {per_expert_diff_mean[idx]} - Element wise diff: {per_expert_element_wise_diff[idx]}"
        )
    with open(os.path.join(save_dir, "diffs.txt"), "w") as f:
        for idx in range(len(args.experts_to_merge)):
            f.write(
                f"Expert {args.experts_to_merge[idx]} - Mean diff: {per_expert_diff_mean[idx]} - Element wise diff: {per_expert_element_wise_diff[idx]}\n"
            )
    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            best_weight.bfloat16(),
            os.path.join(save_dir, "merged_weight.pt"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument(
        "--experts_to_merge", type=str, default="0,1,2,3,4,5,6,7"
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--sample_ratio", type=float, default=0.125)
    parser.add_argument("--inliner_threshold", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert (
        args.weight_id == 1 or args.weight_id == 3
    ), "Weight ID must be in {1, 3}"
    assert (
        args.layer_id >= 0 and args.layer_id <= 31
    ), "Layer ID must be in {0, 1, ..., 31}"

    args.experts_to_merge = list(map(int, args.experts_to_merge.split(",")))

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
