import argparse
import os

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

def get_data(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    print("Loaded data with shape", data.shape, "dtype", data.dtype)
    return data

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
    target_weight = exper_id_to_params[args.expert_id][f"w{args.weight_id}"].to(args.device)
    other_weights = [exper_id_to_params[other_expert_id][f"w{args.weight_id}"].to(args.device) for other_expert_id in range(8) if other_expert_id != args.expert_id]
    other_weights = other_weights[:4]
    # load data
    data = torch.tensor(get_data(args), dtype=_to_torch_dtype(args.dtype), device=args.device)
    # data = data[:16384]
    # calculate image of data
    target_image = target_weight @ data.T
    other_images = [other_weight @ data.T for other_weight in other_weights]
    # weights and data are no longer needed
    del target_weight
    del other_weights
    del data
    # concat other images
    other_images = torch.cat(other_images, axis=0)
    target_image = target_image.to(torch.float).cpu()
    other_images = other_images.to(torch.float).cpu()
    print("Target image shape", target_image.shape, "device", target_image.device, "dtype", target_image.dtype, "taking memory", target_image.element_size() * target_image.numel() / 1024 / 1024, "MB")
    print("Other images shape", other_images.shape, "device", other_images.device, "dtype", other_images.dtype, "taking memory", other_images.element_size() * other_images.numel() / 1024 / 1024, "MB")
    solution = torch.linalg.lstsq(other_images.T, target_image.T, rcond=None)
    linear_map = solution.solution
    print("Linear map shape", linear_map.shape, "residuals:", solution.residuals)
    import code
    code.interact(local=locals())
    exit(0)
    # del data matrices
    del target_image
    del other_images
    # calculate low rank approximation of such linear map
    u, s, v = torch.linalg.svd(linear_map.cpu())
    # plot variance explained
    import matplotlib.pyplot as plt
    import seaborn as sns
    s_squared = s ** 2
    var_explained = s_squared / s_squared.sum()
    var_explained = torch.cumsum(var_explained, dim=0)
    var_explained_cum = var_explained.cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(x=range(1, len(var_explained_cum) + 1), y=var_explained_cum, ax=ax)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Variance explained")
    fig.savefig("linear_map_variance_explained.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.expert_id < 8 and args.expert_id >= 0, "Expert ID must be in [0, 7]"
    assert args.weight_id in [1, 2, 3], "Weight ID must be in [1, 2, 3]"

    main_func(args)

if __name__ == "__main__":
    main()
