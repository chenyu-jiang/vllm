import argparse
import os
from typing import List

import numpy as np
import torch

from lapsolver import solve_dense

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
    for x, w in zip(xs, weights):
        diff_mat = x @ (merged_weight - w.t())
        diff = torch.norm(diff_mat, dim=1)
        per_expert_diff_norm_mean.append(torch.mean(diff).item())
        per_expert_diff_norm_dist.append(list(diff.float().cpu().numpy()))
    return per_expert_diff_norm_mean, per_expert_diff_norm_dist

def get_output_subdir(args, override_weight_id=None):
    if override_weight_id is not None:
        return f"w{override_weight_id}_l{args.layer_id}_esrc{args.expert_to_adapt}_edst{args.expert_to_match}_lambda{args.lamb}" + "_permute" if args.permute else ""
    return f"w{args.weight_id}_l{args.layer_id}_esrc{args.expert_to_adapt}_edst{args.expert_to_match}_lambda{args.lamb}" + "_permute" if args.permute else ""

def get_permutation_output_subdir(args):
    return f"permute_l{args.layer_id}_esrc{args.expert_to_adapt}_edst{args.expert_to_match}"

def train(w_match: torch.Tensor, w_adapt: torch.Tensor, x_match: torch.Tensor, x_adapt: torch.Tensor, lamb=0.5):
    # lambda = 1 -> only match, lambda = 0 -> only adapt

    # compute inner products of xs
    xa_T_xa = x_adapt.t() @ x_adapt
    xm_T_xm = x_match.t() @ x_match
    # reduce non-diagonal items
    # for i in range(len(inner_products)):
    #     inner_products[i] = alpha * inner_products[i] + (1 - alpha) * torch.diag_embed(torch.diagonal(inner_products[i]))
    # compute the inverse of the sum of inner products
    weighted_sum_xT_x = (1 - lamb) * xa_T_xa + lamb * xm_T_xm
    inv_sum_inner_products = torch.inverse(weighted_sum_xT_x.float()).to(x_match.dtype)
    # compute of sum of x^Tx w_i
    xa_T_xa_wa = x_adapt.t() @ x_adapt @ w_adapt.t()
    xm_T_xm_wm = x_match.t() @ x_match @ w_match.t()
    weighted_sum_xT_x_w = (1 - lamb) * xa_T_xa_wa + lamb * xm_T_xm_wm
    # compute the optimal weight
    optimal_weight = inv_sum_inner_products @ weighted_sum_xT_x_w
    return optimal_weight

def solve_permutation_lap(
    w1_match: torch.Tensor,
    w1_permute: torch.Tensor,
    w3_match: torch.Tensor,
    w3_permute: torch.Tensor,
    w2_match: torch.Tensor,
    w2_permute: torch.Tensor,):
    # calculate pairwise cosine similarity
    w1_match_normed = w1_match / torch.norm(w1_match, dim=1, keepdim=True)
    w1_permute_normed = w1_permute / torch.norm(w1_permute, dim=1, keepdim=True)
    w3_match_normed = w3_match / torch.norm(w3_match, dim=1, keepdim=True)
    w3_permute_normed = w3_permute / torch.norm(w3_permute, dim=1, keepdim=True)
    w2_match_normed = w2_match / torch.norm(w2_match, dim=1, keepdim=True)
    w2_permute_normed = w2_permute / torch.norm(w2_permute, dim=1, keepdim=True)

    cost_matrix = w1_permute_normed @ w1_match_normed.t() + w3_permute_normed @ w3_match_normed.t() + w2_permute_normed.t() @ w2_match_normed
    # solve the linear assignment problem
    row_ind, col_ind = solve_dense(cost_matrix.float().cpu().numpy())
    return row_ind, col_ind


def solve_permutation(args, exper_id_to_params: dict):
    # solve permutation first
    w1_match = exper_id_to_params[args.expert_to_match]["w1"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w1_adapt = exper_id_to_params[args.expert_to_adapt]["w1"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w3_match = exper_id_to_params[args.expert_to_match]["w3"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w3_adapt = exper_id_to_params[args.expert_to_adapt]["w3"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w2_match = exper_id_to_params[args.expert_to_match]["w2"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w2_adapt = exper_id_to_params[args.expert_to_adapt]["w2"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    row_ind, col_ind = solve_permutation_lap(w1_match, w1_adapt, w3_match, w3_adapt, w2_match, w2_adapt)
    if not os.path.exists(os.path.join(args.output_dir, get_permutation_output_subdir(args))):
        os.makedirs(os.path.join(args.output_dir, get_permutation_output_subdir(args)), exist_ok=True)
    torch.save((row_ind, col_ind), os.path.join(args.output_dir, get_permutation_output_subdir(args), "permutation_indices.pt"))

def permute_weights(args, w_adapt):
    row_ind, col_ind = torch.load(os.path.join(args.output_dir, get_permutation_output_subdir(args), "permutation_indices.pt"))
    permutation_matrix = torch.zeros((row_ind.shape[0], row_ind.shape[0]), device=args.device, dtype=_to_torch_dtype(args.dtype))
    permutation_matrix[row_ind, col_ind] = 1
    if args.weight_id == 1 or args.weight_id == 3:
        w_adapt = permutation_matrix @ w_adapt
    else:
        w_adapt = w_adapt @ permutation_matrix.t()
    return w_adapt

def create_dataloaders(args):
    data_per_expert = []
    for expert_id in [args.expert_to_match, args.expert_to_adapt]:
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
        data_per_expert.append(data)
    x_match, x_adapt = data_per_expert
    return x_match, x_adapt


def main_func(args, save=True):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                 fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in [args.expert_to_match, args.expert_to_adapt]:
                continue
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    if args.permute and not os.path.exists(os.path.join(args.output_dir, get_permutation_output_subdir(args), "permutation_indices.pt")):
        solve_permutation(args, exper_id_to_params)
    w_match = exper_id_to_params[args.expert_to_match][f"w{args.weight_id}"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w_adapt = exper_id_to_params[args.expert_to_adapt][f"w{args.weight_id}"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    if args.permute:
        w_adapt = permute_weights(args, w_adapt)
    # load data
    x_match, x_adapt = create_dataloaders(args)
    if args.weight_id == 2:
        # generate input to w2 first
        x_match = x_match.bfloat16()
        x_adapt = x_adapt.bfloat16()
        # load in merged weights
        w1_merged = torch.load(os.path.join(args.output_dir, get_output_subdir(args, override_weight_id=1), "merged_weight.pt")).bfloat16()
        w3_merged = torch.load(os.path.join(args.output_dir, get_output_subdir(args, override_weight_id=3), "merged_weight.pt")).bfloat16()
        x_match = torch.nn.functional.silu(x_match @ w1_merged) * (x_match @ w3_merged)
        x_adapt = torch.nn.functional.silu(x_adapt @ w1_merged) * (x_adapt @ w3_merged)
        del w1_merged, w3_merged
        x_match = x_match.to(dtype=_to_torch_dtype(args.dtype))
        x_adapt = x_adapt.to(dtype=_to_torch_dtype(args.dtype))
    # train
    merged_weight = train(w_match, w_adapt, x_match, x_adapt, lamb=args.lamb)
    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(merged_weight.bfloat16(), os.path.join(save_dir, "merged_weight.pt"))
    # evaluate
    per_expert_diff_mean, per_expert_diff_dist = evaluate(merged_weight, [w_match, w_adapt], [x_match, x_adapt])
    print(f"Expert {args.expert_to_match} (To Match) - Mean diff: {per_expert_diff_mean[0]}")
    print(f"Expert {args.expert_to_adapt} (To Adapt) - Mean diff: {per_expert_diff_mean[1]}")
    with open(os.path.join(save_dir, "diffs.txt"), "w") as f:
        f.write(f"Expert {args.expert_to_match} (To Match) - Mean diff: {per_expert_diff_mean[0]}\n")
        f.write(f"Expert {args.expert_to_adapt} (To Adapt) - Mean diff: {per_expert_diff_mean[1]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--expert_to_match", type=int, default=0)
    parser.add_argument("--expert_to_adapt", type=int, default=1)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--permute", action="store_true")
    args = parser.parse_args()

    assert args.weight_id in [1,2,3], "Weight ID must be in {1, 2, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
