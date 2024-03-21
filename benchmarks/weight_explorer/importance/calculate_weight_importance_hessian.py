import argparse
import os

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import tqdm

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

def get_w2_input(w1, w3, X):
    return torch.nn.functional.silu(X @ w1) * (X @ w3)

def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
    return data

def get_output_subdir(args):
    return f"e{args.expert_id}_l{args.layer_id}"

def main_func(args):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1", fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    weights = [exper_id_to_params[args.expert_id][f"w{i}"].to(args.device, dtype=_to_torch_dtype(args.dtype)) for i in range(1, 4)]
    # load data
    data = create_dataloaders(args)
    # calc w1/w3 hessian
    H_w1 = torch.diagonal(2 * data.t() @ data)
    # get w2 input
    w1, w3 = weights[0], weights[2]
    w2_input = get_w2_input(w1.t(), w3.t(), data)
    H_w2 = torch.diagonal(2 * w2_input.t() @ w2_input)

    torch.save((H_w1, H_w2), os.path.join(args.output_dir, get_output_subdir(args), "weight_hessian.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, get_output_subdir(args))):
        os.makedirs(os.path.join(args.output_dir, get_output_subdir(args)), exist_ok=True)

    main_func(args)


if __name__ == "__main__":
    main()
