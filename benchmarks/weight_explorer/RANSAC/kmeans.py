import argparse
import os
from typing import List

from filelock import FileLock

import numpy as np
import torch
import torch.nn.functional as F

from fast_pytorch_kmeans import KMeans

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from tqdm import tqdm

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
        data = torch.tensor(
            data, dtype=_to_torch_dtype(args.dtype), device=args.device
        )
        data_per_expert.append(data)
    return data_per_expert


def main_func(args, save_path=None):
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
    for expert_id in range(8):
      per_expert_weights = []
      for weight_id in range(1, 4):
        per_expert_weights.append(
            exper_id_to_params[expert_id][f"w{weight_id}"]
            .to(args.device)
        )
      weights.append(per_expert_weights)
    # load data
    data = create_dataloaders(args)
    # calculate output per expert
    expert_outputs = []
    for (w1, w2, w3), expert_data in zip(weights, data):
      expert_output = (F.silu(expert_data @ w1.t()) * (expert_data @ w3.t())) @ w2.t()
      expert_outputs.append(expert_output)

    mean_distances = []
    mean_norms = []
    for expert_output in tqdm(expert_outputs):
      kmeans = KMeans(n_clusters=args.n_clusters)
      labels = kmeans.fit_predict(expert_output)
      centroids = kmeans.centroids
      # calculate distance to centroids
      centroids_selected = centroids[labels]
      distance = torch.norm(expert_output - centroids_selected, dim=1).mean().item()
      mean_distances.append(distance)
      mean_norms.append(torch.norm(expert_output, dim=1).mean().item())
    if args.save_path is not None:
      lock = FileLock(args.save_path + ".lock")
      with lock:
        if not os.path.exists(args.save_path):
          with open(args.save_path, "w") as f:
            f.write("n_clusters,layer_id,expert_id,mean_distance,mean_norm\n")
        with open(args.save_path, "a") as f:
          for expert_id, distance, mean_norm in zip(range(8), mean_distances, mean_norms):
            f.write(f"{args.n_clusters},{args.layer_id},{expert_id},{distance},{mean_norm}\n")
    else:
      for expert_id, distance, mean_norm in zip(range(8), mean_distances, mean_norms):
        print(f"Expert {expert_id} mean distance: {distance}, mean norm: {mean_norm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./kmeans_results.csv")
    args = parser.parse_args()

    assert (
        args.layer_id >= 0 and args.layer_id <= 31
    ), "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
