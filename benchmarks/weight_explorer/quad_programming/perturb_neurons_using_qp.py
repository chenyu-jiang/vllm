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


def get_output_subdir(args, override_weight_id=None):
    if override_weight_id is not None:
        return f"w{override_weight_id}_l{args.layer_id}_e{args.expert_id}"
    return f"w{args.weight_id}_l{args.layer_id}_e{args.expert_id}"



def solve_qp(
    neurons_to_replace: torch.Tensor,  # shape (1, d) or (b, d)
    neurons_to_perturb: torch.Tensor,  # shape (b, d)
    x0: torch.Tensor,  # shape (d, n)
    xi: torch.Tensor,  # shape (d, n)
    norm_limit: float,
):
    if len(neurons_to_perturb.shape) == 1:
        neurons_to_replace = neurons_to_replace.unsqueeze(0)

    assert neurons_to_perturb.shape[1] == neurons_to_replace.shape[1]
    assert neurons_to_perturb.shape[0] == neurons_to_replace.shape[0] or neurons_to_replace.shape[0] == 1

    def objective(result_neuron):
        # (neuron_to_replace - result_neuron) @ x0 shape: (b, n)
        return (
            torch.norm((neurons_to_replace - result_neuron) @ x0, dim=-1) ** 2
        )

    print("Norm limit: ", norm_limit)

    def constraint(result_neuron):
        # (result_neuron - neurons_to_perturb) @ xi shape: (b, n)
        return (
            (torch.norm((result_neuron - neurons_to_perturb) @ xi, dim=-1) ** 2) / xi.shape[1]
            - norm_limit
        )

    # init result_neuron to be neuron_to_perturb
    result_neuron = neurons_to_perturb.clone()
    # solve using interior point method
    result_neuron = interior_point(objective, constraint, result_neuron)
    import code
    code.interact(local=locals())
    final_objective = objective(result_neuron)
    final_constraint = constraint(result_neuron)
    return result_neuron, final_objective, final_constraint


def train(
    current_neuron: torch.Tensor,
    current_expert_x: torch.Tensor,
    other_expert_ids: List[int],
    other_expert_neurons: List[torch.Tensor],
    other_expert_xs: List[torch.Tensor],
    norm_limit: float,
    euc_dist_quantile_filter: float,
    batch_size: int,
):
    if len(current_neuron.shape) == 1:
        # make sure current_neuron is broadcastable
        current_neuron = current_neuron.unsqueeze(0)

    mean_reconstruct_errors = []
    mean_perturbation_errors = []
    expert_ids = []
    min_reconstruct_error = float("inf")
    best_perturbed_neuron = None
    for expert_id, this_expert_xs, this_expert_neurons in tqdm(
        zip(other_expert_ids, other_expert_xs, other_expert_neurons),
        total=len(other_expert_xs),
        desc="Experts",
    ):
        euclidean_distances = torch.norm(
            current_neuron - this_expert_neurons, dim=-1
        )
        _, indices = torch.sort(euclidean_distances)
        # keep top euc_dist_quantile_filter neurons
        this_expert_neurons = this_expert_neurons[indices[: int(len(indices) * euc_dist_quantile_filter)]]
        n_batches = (
            this_expert_neurons.shape[0] + batch_size - 1
        ) // batch_size
        for i in trange(n_batches, desc="Neuron Batch", leave=False):
            this_batch_neurons = this_expert_neurons[
                i * batch_size : (i + 1) * batch_size
            ]
            perturbed_neuron = solve_qp(
                current_neuron,
                this_batch_neurons,
                current_expert_x,
                this_expert_xs,
                (norm_limit * norm_limit) * this_expert_xs.shape[1],
            )
            # calculate error
            reconstruct_error = (
                torch.norm(
                    (current_neuron - perturbed_neuron) @ current_expert_x,
                    dim=-1,
                )
                / current_expert_x.shape[1]
            )
            perturbation_error = (
                torch.norm(
                    (perturbed_neuron - this_batch_neurons) @ this_expert_xs,
                    dim=-1,
                )
                / this_expert_xs.shape[1]
            )
            mean_reconstruct_errors.extend(reconstruct_error.tolist())
            mean_perturbation_errors.extend(perturbation_error.tolist())
            expert_ids.append(expert_id)
            min_error, min_idx = reconstruct_error.min(dim=0)
            if min_error < min_reconstruct_error:
                min_reconstruct_error = min_error
                best_perturbed_neuron = perturbed_neuron[min_idx]
    return mean_reconstruct_errors, mean_perturbation_errors, expert_ids, best_perturbed_neuron

def train_with_most_similar_hueristic(
    current_neurons: torch.Tensor,
    current_expert_x: torch.Tensor,
    other_expert_ids: List[int],
    other_expert_neurons: List[torch.Tensor],
    other_expert_xs: List[torch.Tensor],
    norm_limit: float,
    batch_size: int,
):
    concated_other_expert_neurons = torch.cat(other_expert_neurons, dim=0)
    # find the most similar neurons for each current neuron
    pairwise_euc_dist = torch.cdist(current_neurons, concated_other_expert_neurons)
    _, most_similar_indices = torch.min(pairwise_euc_dist, dim=1)
    most_similar_indices = most_similar_indices.tolist()
    # group most similar neuron by expert id
    index_map = {}
    expert_map = {}
    most_similar_neurons_per_expert = {}
    for i, neuron_id in enumerate(most_similar_indices):
        expert_id = other_expert_ids[neuron_id // other_expert_neurons[0].shape[0]]
        expert_map[i] = expert_id
        if expert_id not in most_similar_neurons_per_expert:
            most_similar_neurons_per_expert[expert_id] = {"to_remove": [], "to_perturb": []}
        most_similar_neurons_per_expert[expert_id]["to_perturb"].append(concated_other_expert_neurons[neuron_id])
        most_similar_neurons_per_expert[expert_id]["to_remove"].append(current_neurons[i])
        index_map[i] = len(most_similar_neurons_per_expert[expert_id]["to_perturb"]) - 1

    for expert_id in most_similar_neurons_per_expert:
        most_similar_neurons_per_expert[expert_id]["to_perturb"] = torch.stack(most_similar_neurons_per_expert[expert_id]["to_perturb"])
        most_similar_neurons_per_expert[expert_id]["to_remove"] = torch.stack(most_similar_neurons_per_expert[expert_id]["to_remove"])

    perturbed_neuron_per_expert = {}
    for expert_id, this_expert_xs in tqdm(
        zip(other_expert_ids, other_expert_xs),
        total=len(other_expert_xs),
        desc="Experts",
    ):
        if expert_id not in most_similar_neurons_per_expert:
            continue
        this_expert_neurons_to_perturb = most_similar_neurons_per_expert[expert_id]["to_perturb"]
        this_expert_neurons_to_remove = most_similar_neurons_per_expert[expert_id]["to_remove"]
        n_batches = (
            this_expert_neurons_to_perturb.shape[0] + batch_size - 1
        ) // batch_size
        batch_outs = []
        batch_objectives = []
        batch_constraints = []
        for i in trange(n_batches, desc="Neuron Batch", leave=False):
            this_batch_neurons_to_perturb = this_expert_neurons_to_perturb[
                i * batch_size : (i + 1) * batch_size
            ]
            this_batch_neurons_to_remove = this_expert_neurons_to_remove[
                i * batch_size : (i + 1) * batch_size
            ]
            perturbed_neuron, final_objective, final_constraint = solve_qp(
                this_batch_neurons_to_remove,
                this_batch_neurons_to_perturb,
                current_expert_x,
                this_expert_xs,
                (norm_limit * norm_limit)
            )
            batch_outs.append(perturbed_neuron)
            batch_objectives.append(final_objective)
            batch_constraints.append(final_constraint)
        mean_objective = torch.concat(batch_objectives).mean().item()
        max_constraint = torch.concat(batch_constraints).max().item()
        print(f"Expert {expert_id}: Mean Objective: {mean_objective}, Max Constraint: {max_constraint}")
        perturbed_neuron_per_expert[expert_id] = torch.cat(batch_outs, dim=0)
    # reorder
    result = []
    for i in range(len(current_neurons)):
        result.append(perturbed_neuron_per_expert[expert_map[i]][index_map[i]])
    result = torch.stack(result)
    return result

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
    weights = []
    for expert_id in range(8):
        weights.append(
            exper_id_to_params[expert_id][f"w{args.weight_id}"]
            .to(args.device)
        )
    if args.weight_id == 2:
        w1_weights = []
        w3_weights = []
        for expert_id in range(8):
            w1_weights.append(
                exper_id_to_params[expert_id][f"w1"]
                .to(args.device)
            )
            w3_weights.append(
                exper_id_to_params[expert_id][f"w3"]
                .to(args.device)
            )
    # load data
    data = create_dataloaders(args)

    current_neurons = weights[args.expert_id]
    # load perturbed w1 and w3 if they exist
    if args.weight_id == 2:
        perturbed_w1 = torch.load(
            os.path.join(
                args.output_dir,
                get_output_subdir(args, override_weight_id=1),
                "results.pt",
            )
        )
        perturbed_w3 = torch.load(
            os.path.join(
                args.output_dir,
                get_output_subdir(args, override_weight_id=3),
                "results.pt",
            )
        )
        perturbed_w1 = perturbed_w1.to(device=args.device).bfloat16()
        perturbed_w3 = perturbed_w3.to(device=args.device).bfloat16()
        # obtain w2 input
        current_expert_x = F.silu(data[args.expert_id] @ perturbed_w1.t()) * (data[args.expert_id] @ perturbed_w3.t())
        current_expert_x = current_expert_x.t()
    else:
        current_expert_x = data[args.expert_id].t()
    other_expert_ids = [x for x in list(range(8)) if x != args.expert_id]
    other_expert_neurons = [weights[x] for x in other_expert_ids]
    if args.weight_id == 2:
        # obtain w2 input
        other_expert_xs = [
            (F.silu(data[x] @ w1_weights[x].t()) * (data[x] @ w3_weights[x].t())).t()
            for x in other_expert_ids
        ]
        del w1_weights
        del w3_weights
    else:
        other_expert_xs = [data[x].t() for x in other_expert_ids]
    perturbed_neurons = train_with_most_similar_hueristic(
        current_neurons,
        current_expert_x,
        other_expert_ids,
        other_expert_neurons,
        other_expert_xs,
        args.norm_limit,
        args.batch_size,
    )
    # save
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            perturbed_neurons,
            os.path.join(save_dir, "results.pt"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--norm_limit", type=float, default=1e-4)
    parser.add_argument("--euc_dist_quantile_filter", type=float, default=0.05)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert (
        args.weight_id in [1, 2, 3]
    ), "Weight ID must be in {1, 2, 3}"
    assert (
        args.layer_id >= 0 and args.layer_id <= 31
    ), "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
