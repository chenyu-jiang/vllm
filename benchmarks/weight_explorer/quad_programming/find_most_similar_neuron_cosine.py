import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from interior_point import interior_point

from tqdm import tqdm, trange


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_output_subdir(args):
    return f"l{args.layer_id}_e{args.expert_id}"

def evaluate(data, orig_w1, orig_w2, orig_w3, reconstructed_w1, reconstructed_w2, reconstructed_w3):
    orig_out = (F.silu(data @ orig_w1.t()) * (data @ orig_w3.t())) @ orig_w2.t()
    reconstructed_out = (F.silu(data @ reconstructed_w1.t()) * (data @ reconstructed_w3.t())) @ reconstructed_w2.t()
    diff = orig_out - reconstructed_out
    return torch.norm(diff, dim=-1).mean().item()

def find_most_similar_neuron_cosine(
    weight_to_replace: torch.Tensor,
    candidate_weights: List[torch.Tensor]):
    # calculate pairwise cosine similarity
    concatenated_weights = torch.cat(candidate_weights, dim=0)
    concatenated_weights = concatenated_weights / torch.norm(concatenated_weights, dim=-1, keepdim=True)
    weight_to_replace = weight_to_replace / torch.norm(weight_to_replace, dim=-1, keepdim=True)
    cosine_similarities = torch.mm(weight_to_replace, concatenated_weights.t())
    # find the most similar neuron
    max_sim, max_idx = torch.max(cosine_similarities, dim=1)
    # return the most similar neuron (with unit norm)
    new_weight = concatenated_weights[max_idx]
    # scale back
    new_weight = new_weight * torch.norm(weight_to_replace, dim=-1, keepdim=True)
    return max_sim, new_weight

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
    w1_weights = []
    w2_weights = []
    w3_weights = []
    for expert_id in range(8):
        w1_weights.append(
            exper_id_to_params[expert_id][f"w1"]
            .to(args.device)
        )
        w2_weights.append(
            exper_id_to_params[expert_id][f"w2"]
            .to(args.device)
        )
        w3_weights.append(
            exper_id_to_params[expert_id][f"w3"]
            .to(args.device)
        )
    # load data
    data = create_dataloaders(args)
    # train
    print("Finding the most similar neuron for w2...")
    w1_sim, new_w1 = find_most_similar_neuron_cosine(
        w1_weights[args.expert_id],
        [w for idx, w in enumerate(w1_weights) if idx != args.expert_id]
    )
    print("Finding the most similar neuron for w2...")
    w2_sim, new_w2 = find_most_similar_neuron_cosine(
        w2_weights[args.expert_id],
        [w for idx, w in enumerate(w2_weights) if idx != args.expert_id]
    )
    print("Finding the most similar neuron for w2...")
    w3_sim, new_w3 = find_most_similar_neuron_cosine(
        w3_weights[args.expert_id],
        [w for idx, w in enumerate(w3_weights) if idx != args.expert_id]
    )
    # evaluate
    diff = evaluate(data[args.expert_id], w1_weights[args.expert_id], w2_weights[args.expert_id], w3_weights[args.expert_id], new_w1, new_w2, new_w3)
    # save
    if save:
        output_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(output_dir, exist_ok=True)
        torch.save((new_w1, new_w2, new_w3), os.path.join(output_dir, "new_weights.pt"))
        torch.save((w1_sim, w2_sim, w3_sim), os.path.join(output_dir, "cosine_similarities.pt"))
        with open(os.path.join(output_dir, "diff.txt"), "w") as f:
            f.write(f"Mean diff: {diff}")
    import code
    code.interact(local=locals())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert (
        args.layer_id >= 0 and args.layer_id <= 31
    ), "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
