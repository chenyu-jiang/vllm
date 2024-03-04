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

def layer_func(w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor, x: torch.Tensor):
    w1o = torch.nn.functional.silu(x @ w1.t())
    w3o = x @ w3.t()
    return (w1o * w3o) @ w2.t()

def create_dataloaders(args):
    data_per_expert = []
    for expert_id in range(8):
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
        data_per_expert.append(data)
    return data_per_expert

def main_func(args):
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
    w2_weights = []
    for expert_id in range(8):
        w2_weights.append(exper_id_to_params[expert_id][f"w2"].to(args.device))
        w1_weights.append(exper_id_to_params[expert_id][f"w1"].to(args.device))
        w3_weights.append(exper_id_to_params[expert_id][f"w3"].to(args.device))
    # load data
    data = create_dataloaders(args)
    # calculate outputs
    for expert_id in range(8):
        x = data[expert_id]
        w1 = w1_weights[expert_id]
        w3 = w3_weights[expert_id]
        w2 = w2_weights[expert_id]
        output = layer_func(w1, w3, w2, x)
        output_mean_norm = torch.mean(torch.norm(output, dim=1))
        if not os.path.exists(args.output_path):
            with open(args.output_path, "w") as f:
                f.write("expert_id,layer_id,output_mean_norm\n")
        with open(args.output_path, "a") as f:
            f.write(f"{expert_id},{args.layer_id},{output_mean_norm}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
