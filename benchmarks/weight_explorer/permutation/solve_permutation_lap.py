import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from lapsolver import solve_dense

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_output_subdir(args):
    return f"l{args.layer_id}_e{args.expert_id_to_permute}"

def evaluate(data, orig_w1, orig_w2, orig_w3, permuted_w1, permuted_w2, permuted_w3):
    orig_out = (F.silu(data @ orig_w1.t()) * (data @ orig_w3.t())) @ orig_w2.t()
    reconstructed_out = (F.silu(data @ permuted_w1.t()) * (data @ permuted_w3.t())) @ permuted_w2.t()
    diff = orig_out - reconstructed_out
    return torch.norm(diff, dim=-1).mean().item()

def solve_permutation_lap(
    w1_match: torch.Tensor,
    w1_permute: torch.Tensor,
    w3_match: torch.Tensor,
    w3_permute: torch.Tensor,
    w2_match: torch.Tensor,
    w2_permute: torch.Tensor,):
    # calculate pairwise cosine similarity
    cost_matrix = w1_permute @ w1_match.t() + w3_permute @ w3_match.t() + w2_permute.t() @ w2_match
    # solve the linear assignment problem
    row_ind, col_ind = solve_dense(cost_matrix.float().cpu().numpy())
    return row_ind, col_ind

def create_dataloaders(args):
    data_per_expert = []
    for expert_id in range(8):
        data = np.load(
            os.path.join(
                args.data_dir,
                f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz",
            )
        )["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=torch.bfloat16, device=args.device)
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
    w1_match: torch.Tensor = exper_id_to_params[args.expert_id_to_match]["w1"].to(args.device)
    w1_permute: torch.Tensor = exper_id_to_params[args.expert_id_to_permute]["w1"].to(args.device)
    w3_match: torch.Tensor = exper_id_to_params[args.expert_id_to_match]["w3"].to(args.device)
    w3_permute: torch.Tensor = exper_id_to_params[args.expert_id_to_permute]["w3"].to(args.device)
    w2_match: torch.Tensor = exper_id_to_params[args.expert_id_to_match]["w2"].to(args.device)
    w2_permute: torch.Tensor = exper_id_to_params[args.expert_id_to_permute]["w2"].to(args.device)
    # load data
    # data = create_dataloaders(args)
    # train
    row_ind, col_ind = solve_permutation_lap(w1_match, w1_permute, w3_match, w3_permute, w2_match, w2_permute)
    # construct full matrix
    perm_matrix = torch.zeros((w1_permute.shape[0], w1_permute.shape[0]), device=args.device, dtype=torch.bfloat16)
    perm_matrix[row_ind, col_ind] = 1
    # permute
    permuted_w1 = perm_matrix @ w1_permute
    permuted_w2 = w2_permute @ perm_matrix.t()
    permuted_w3 = perm_matrix @ w3_permute
    # save results
    if save:
        # save permuted weights
        subdir = get_output_subdir(args)
        full_dir_path = os.path.join(args.output_dir, subdir)
        os.makedirs(full_dir_path, exist_ok=True)
        torch.save((permuted_w1, permuted_w2, permuted_w3), os.path.join(full_dir_path, "permuted_weights.pt"))
        print(f"Saved permuted weights to {full_dir_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expert_id_to_match", type=int, default=0)
    parser.add_argument("--expert_id_to_permute", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    # parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert (
        args.layer_id >= 0 and args.layer_id <= 31
    ), "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)

if __name__ == "__main__":
    main()