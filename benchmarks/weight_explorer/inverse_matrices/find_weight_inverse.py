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
    for x, w in zip(xs, weights):
        diff_mat = x @ (merged_weight - w.t())
        diff = torch.norm(diff_mat, dim=1)
        per_expert_diff_norm_mean.append(torch.mean(diff).item())
        per_expert_diff_norm_dist.append(list(diff.float().cpu().numpy()))
    return per_expert_diff_norm_mean, per_expert_diff_norm_dist

def get_output_subdir(args, override_weight_id=None):
    if override_weight_id is not None:
        return f"w{override_weight_id}_l{args.layer_id}_etgt{args.expert_id}_ecandidates{','.join([str(x) for x in args.candidate_experts])}"
    return f"w{args.weight_id}_l{args.layer_id}_etgt{args.expert_id}_ecandidates{','.join([str(x) for x in args.candidate_experts])}"

def find_inverse(w_target: torch.Tensor, w_candidates: List[torch.Tensor]):
    concated_candidates = torch.cat(w_candidates, dim=0)
    wTw = concated_candidates.t() @ concated_candidates
    wTw = wTw.float()
    inv_wTw = torch.inverse(wTw).to(w_target.dtype)
    return w_target @ inv_wTw @ concated_candidates.t()

def main_func(args, save=True):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                 fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in [args.expert_id] + args.candidate_experts:
                continue
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    w_target = exper_id_to_params[args.expert_id][f"w{args.weight_id}"].to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    w_candidates = [exper_id_to_params[expert_id][f"w{args.weight_id}"] for expert_id in args.candidate_experts]
    w_candidates = [w.to(device=args.device, dtype=_to_torch_dtype(args.dtype)) for w in w_candidates]
    # train
    inverse_map = find_inverse(w_target, w_candidates)
    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(inverse_map, os.path.join(save_dir, "inverse_map.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--candidate_experts", type=str, default="1,2,3,4,5,6,7")
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    args.candidate_experts = [int(x) for x in args.candidate_experts.split(",")]

    assert args.weight_id in [1,2,3], "Weight ID must be in {1, 2, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
