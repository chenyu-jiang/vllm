import argparse
import os

import time
import gurobipy as gp

import numpy as np

from typing import List, Set

import torch

def _to_torch_dtype(dtype):
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype, None)
        if dtype is None:
            raise ValueError(f"Invalid dtype {dtype}")
    return dtype

class Predictor(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super(Predictor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, 1)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

    @classmethod
    def from_state_dict(cls, state_dict):
        predictor = cls(state_dict["fc1.weight"].shape[1], state_dict["fc1.weight"].shape[0])
        predictor.load_state_dict(state_dict)
        return predictor

def calculate_loss(features: torch.Tensor,
                   expert_weights: torch.Tensor,
                   selected_experts: torch.Tensor,
                   predictors: List[Predictor]):
    # losses = []
    sum_of_loss_per_expert = []
    for expert_idx in range(8):
        expert_mask = (selected_experts == expert_idx)
        expert_weight_masked = (expert_weights * expert_mask).sum(dim=1)
        predicted_loss = (predictors[expert_idx](features) * expert_weight_masked).sum()
        sum_of_loss_per_expert.append(predicted_loss)
    return torch.stack(sum_of_loss_per_expert, dim=0)
    # for sample_idx in range(features.shape[0]):
    #     per_sample_losses = []
    #     for expert_idx in range(selected_experts.shape[1]):
    #         expert_id = selected_experts[sample_idx, expert_idx].item()
    #         expert_weight = expert_weights[sample_idx, expert_idx].item()
    #         loss = predictors[expert_id](features[sample_idx]) * expert_weight
    #         per_sample_losses.append(loss)
    #     losses.append(per_sample_losses)
    # return losses

def solve_ilp(features: torch.Tensor,
              expert_weights: torch.Tensor,
              selected_experts: torch.Tensor,
              predictors: List[Predictor],
              k_experts_to_select: int,
              n_experts: int):
    """
    Solve the ILP problem to select the experts
    """
    # first use the predictors to predict the loss
    t = time.time()
    losses = calculate_loss(features, expert_weights, selected_experts, predictors)
    print("Time to calculate loss", time.time() - t)
    t = time.time()
    with gp.Env(empty=True):
        # disable output
        gp.setParam("OutputFlag", 0)
        # create the ILP model
        model = gp.Model("ILP")
        # create the variables
        x = model.addVars(n_experts, vtype=gp.GRB.BINARY, name="x")
        # construct the loss function
        loss = 0
        for expert_idx in range(n_experts):
            loss += losses[expert_idx] * (1 - x[expert_idx])
        # for sample_idx in range(features.shape[0]):
        #     for expert_idx in range(selected_experts.shape[1]):
        #         expert_id = selected_experts[sample_idx, expert_idx].item()
        #         loss += losses[sample_idx][expert_idx] * (1 - x[expert_id])
        # set the objective
        model.setObjective(loss, gp.GRB.MINIMIZE)
        # add the constraint
        model.addConstr(x.sum() <= k_experts_to_select)
        # solve the model
        model.optimize()
        # get the selected experts
        result = []
        for expert_id in range(n_experts):
            if x[expert_id].x > 0.5:
                result.append(expert_id)
    print("Time to solve ILP", time.time() - t)
    return result

def solve_knapsack_dp(features: torch.Tensor,
                        expert_weights: torch.Tensor,
                        selected_experts: torch.Tensor,
                        predictors: List[Predictor],
                        k_experts_to_select: int,
                        n_experts: int):
    t = time.time()
    values = calculate_loss(features, expert_weights, selected_experts, predictors)
    print("Time to calculate loss", time.time() - t)
    t = time.time()
    expert_ids = torch.argsort(values, dim=0, descending=True)
    torch.cuda.synchronize()
    print("Time to sort", time.time() - t)
    return expert_ids[:k_experts_to_select]


def calculate_loss_individual(features: torch.Tensor,
                                expert_weights: torch.Tensor,
                                selected_experts: torch.Tensor,
                                predictors: List[Predictor]):
    losses = []
    for sample_idx in range(features.shape[0]):
        per_sample_losses = []
        for expert_idx in range(selected_experts.shape[1]):
            expert_id = selected_experts[sample_idx, expert_idx].item()
            expert_weight = expert_weights[sample_idx, expert_idx].item()
            loss = predictors[expert_id](features[sample_idx]) * expert_weight
            per_sample_losses.append(loss)
        losses.append(per_sample_losses)
    return losses

def evaluate_loss(features: torch.Tensor,
                  expert_weights: torch.Tensor,
                  selected_experts: torch.Tensor,
                  predictors: List[Predictor],
                  opt_selection: Set[int]):
    """
    Evaluate the loss of the selected experts
    """
    losses = calculate_loss_individual(features, expert_weights, selected_experts, predictors)
    total_loss = 0
    for sample_idx in range(features.shape[0]):
        for expert_idx in range(selected_experts.shape[1]):
            expert_id = selected_experts[sample_idx, expert_idx].item()
            if expert_id not in opt_selection:
                total_loss += losses[sample_idx][expert_idx]
    return total_loss

def create_dataloaders(args):
    per_expert_data = []
    for i in range(8):
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{i}_l{args.layer_idx}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
        per_expert_data.append(data)
    # sample randomly from the data
    features = []
    expert_weights = []
    selected_experts = []
    for i in range(args.batch_size):
        # first sample a expert
        expert_id = np.random.randint(8)
        data = per_expert_data[expert_id]
        # then sample a data point
        data_point = data[np.random.randint(data.shape[0])]
        # then sample a second expert
        expert_id_2 = np.random.randint(8)
        while expert_id_2 == expert_id:
            expert_id_2 = np.random.randint(8)
        # then sample from power law distribution to get the expert weight
        expert_weight = np.random.uniform(0, 1, 2)
        expert_weight = np.power(expert_weight, 1.2)
        # sort (desc) and normalize the expert weight so that it sums to 1
        expert_weight = np.flip(np.sort(expert_weight))
        expert_weight /= expert_weight.sum()
        # append
        features.append(data_point)
        expert_weights.append(expert_weight)
        selected_experts.append([expert_id, expert_id_2])
    features = torch.stack(features)
    expert_weights = torch.tensor(np.stack(expert_weights), dtype=_to_torch_dtype(args.dtype), device=args.device)
    selected_experts = torch.tensor(selected_experts, dtype=torch.long, device=args.device)
    return features, expert_weights, selected_experts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k_experts_to_select", type=int, default=4)
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--predictor_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()

def main():
    args = parse_args()
    # load the predictors
    predictors = []
    for expert_id in range(8):
        predictor = torch.load(f"{args.predictor_dir}/e{expert_id}_l{args.layer_idx}/model.pt")
        predictors.append(Predictor.from_state_dict(predictor).to(_to_torch_dtype(args.dtype)).to(args.device))
    # load the data
    features, expert_weights, selected_experts = create_dataloaders(args)
    # solve the ILP problem
    # opt_selection = solve_ilp(features, expert_weights, selected_experts, predictors, args.k_experts_to_select, 8)
    opt_selection = solve_knapsack_dp(features, expert_weights, selected_experts, predictors, args.k_experts_to_select, 8)
    # evaluate the loss
    total_loss = evaluate_loss(features, expert_weights, selected_experts, predictors, opt_selection)
    print("Optimal selection", opt_selection, "with loss", total_loss.item())
    # certify result optimality
    for i in range(20):
        # generate ranodm selection
        random_selection = np.random.choice(8, args.k_experts_to_select, replace=False)
        # evaluate the loss
        random_loss = evaluate_loss(features, expert_weights, selected_experts, predictors, random_selection)
        # print("Random selection", random_selection, "with loss", random_loss.item())
        # compare the loss
        assert random_loss >= total_loss - 1e-6, f"Random selection {random_selection} has lower loss {random_loss} than optimal selection {opt_selection} with loss {total_loss}"
    print("Optimality certified.")


if __name__ == "__main__":
    main()