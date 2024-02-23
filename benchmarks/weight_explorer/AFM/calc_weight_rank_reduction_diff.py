import argparse
import os

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import tqdm

from vllm.model_executor.weight_utils import hf_model_weights_iterator

def to_torch_dtype(t, dtype, device):
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype, None)
        if dtype is None:
            raise ValueError(f"Invalid dtype {dtype}")
    return t.to(dtype).to(device)

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def calc_norm_diff_after_rank_reduction(w1, w2, w3, w1_dst, w2_dst, w3_dst, ranks, dtype, device):
    norm_diffs = defaultdict(list)
    norm_diff_percents = defaultdict(list)
    mean_diffs = defaultdict(list)
    max_diffs = defaultdict(list)
    explained_vars = defaultdict(list)
    all_outputs = {"w1": to_torch_dtype(w1_dst - w1, dtype, device),
                   "w2": to_torch_dtype(w2_dst - w2, dtype, device),
                     "w3": to_torch_dtype(w3_dst - w3, dtype, device)}
    for w_name in tqdm.tqdm(["w1", "w3", "w2"], desc="Weight"):
        concat_output = all_outputs[w_name]
        # calculate SVD
        U, S, Vh = torch.linalg.svd(concat_output)
        S_sq = S ** 2
        for rank in tqdm.tqdm(ranks, desc="Rank"):
            reduced_output = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            norm_diff = torch.linalg.norm(concat_output - reduced_output)
            norm_diff_percent = norm_diff / torch.linalg.norm(concat_output)
            mean_diff = torch.mean(torch.abs(concat_output - reduced_output))
            max_diff = torch.max(torch.abs(concat_output - reduced_output))
            explained_var = S_sq[:rank].sum() / S_sq.sum()
            norm_diffs[w_name].append(norm_diff.item())
            norm_diff_percents[w_name].append(norm_diff_percent.item())
            mean_diffs[w_name].append(mean_diff.item())
            max_diffs[w_name].append(max_diff.item())
            explained_vars[w_name].append(explained_var.item())
    return norm_diffs, norm_diff_percents, mean_diffs, max_diffs, explained_vars

def main_func(args):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1"):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    src_weights = [exper_id_to_params[args.src_expert_id][f"w{i}"] for i in range(1, 4)]
    dst_weights = [exper_id_to_params[args.dst_expert_id][f"w{i}"] for i in range(1, 4)]
    # calculate norm diff
    ranks = [8, 16, 32, 64, 128, 256, 512] + list(range(1024, 4096 + 512, 512))
    norm_diffs, norm_diff_percents, mean_diffs, max_diffs, explained_vars = calc_norm_diff_after_rank_reduction(*src_weights, *dst_weights, ranks, args.dtype, args.device)
    # save results
    exists = os.path.exists(args.output_path)
    with open(args.output_path, "a") as f:
        if not exists:
            f.write("weight,src_expert_id,dst_expert_id,layer_id,rank,norm_diff,norm_lost_percent,mean_diff,max_diff,explained_var\n")
        for w_name in ["w1", "w3", "w2"]:
            for rank, norm_diff, norm_diff_percent, mean_diff, max_diff, explained_var in zip(ranks, norm_diffs[w_name], norm_diff_percents[w_name], mean_diffs[w_name], max_diffs[w_name], explained_vars[w_name]):
                f.write(f"{w_name},{args.src_expert_id},{args.dst_expert_id},{args.layer_id},{rank},{norm_diff},{1 - norm_diff_percent},{mean_diff},{max_diff},{explained_var}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--src_expert_id", type=int, default=0)
    parser.add_argument("--dst_expert_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    assert args.src_expert_id != args.dst_expert_id, "src_expert_id and dst_expert_id should be different"

    main_func(args)


if __name__ == "__main__":
    main()
