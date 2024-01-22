import random
import os
from collections import defaultdict
from typing import List, Tuple

import argparse

import tqdm

import gurobipy as gp

def calculate_activated_experts(expert_selection_per_node: List[Tuple[int, int]],
                                selected_indices: List[int]) -> int:
    activated_experts = set()
    for i in selected_indices:
        for exp in expert_selection_per_node[i]:
            activated_experts.add(exp)
    return len(activated_experts)

def create_is_gt0_integer_var(model: gp.Model, x: gp.Var, big_M: float = 1000):
    """
    Create a variable that is 1 if x > 0, 0 otherwise.
    """
    z = model.addVar(vtype=gp.GRB.BINARY)
    model.addConstr(x <= big_M * z)
    return z

def find_best_tokens(expert_selection_per_node: List[Tuple[int, int]],
                   max_batch_size: int
                   ) -> int:
    # create optimization variables
    # x_i = 1 if request i should be scheduled in the batch
    if max_batch_size >= len(expert_selection_per_node):
        best_req_indices = list(range(len(expert_selection_per_node)))
        activated_experts = calculate_activated_experts(expert_selection_per_node, best_req_indices)
        return activated_experts
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        # env.setParam("MIPGap", 0.2)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(len(expert_selection_per_node), vtype=gp.GRB.BINARY)
            per_expert_vars = {}
            for i, expert_selection in enumerate(expert_selection_per_node):
                for exp in expert_selection:
                    if exp not in per_expert_vars:
                        per_expert_vars[exp] = []
                    per_expert_vars[exp].append(x[i])
            # objective: minimize number of experts activated
            expert_activation_indicators = {}
            for exp_id, variables in per_expert_vars.items():
                expert_activation_indicators[exp_id]= create_is_gt0_integer_var(model, gp.quicksum(variables), len(variables) + 1)
            # objective: minimize number of experts activated
            model.setObjective(gp.quicksum(expert_activation_indicators.values()), gp.GRB.MINIMIZE)
            # constraint: sum of x_i <= max_batch_size
            model.addConstr(x.sum() == max_batch_size)
            model.addConstr(x >= 0)
            model.addConstr(x <= 1)
            model.optimize()
            # get top max_batch_size indices
            xis = [(i, x[i].X) for i in range(len(expert_selection_per_node))]
            xis = sorted(xis, key=lambda x: x[1], reverse=True)
            best_req_indices = [xi[0] for xi in xis[:max_batch_size]]
    activated_experts = calculate_activated_experts(expert_selection_per_node, best_req_indices)
    return activated_experts


def find_best_tokens_and_experts(expert_selection_per_node: List[Tuple[int, ...]],
                                 k_experts_per_token: int,
                                 max_batch_size: int,
                                 ) -> int:
    n_candidate_experts = len(expert_selection_per_node[0])
    if n_candidate_experts == k_experts_per_token:
        return find_best_tokens(expert_selection_per_node, max_batch_size)
    # create optimization variables
    # x_i,j = 1 if request i should be scheduled in the batch and
    # expert j should be activated (besides the first expert)
    if max_batch_size >= len(expert_selection_per_node):
        max_batch_size = len(expert_selection_per_node)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar((len(expert_selection_per_node), n_candidate_experts), vtype=gp.GRB.BINARY)
            per_expert_vars = {}
            for i, expert_selection in enumerate(expert_selection_per_node):
                for j, exp in enumerate(expert_selection):
                    if exp not in per_expert_vars:
                        per_expert_vars[exp] = []
                    per_expert_vars[exp].append(x[i][j])
            # objective: minimize number of experts activated
            expert_activation_indicators = {}
            for exp_id, variables in per_expert_vars.items():
                expert_activation_indicators[exp_id]= create_is_gt0_integer_var(model, gp.quicksum(variables), len(variables) + 1)
            # objective: minimize number of experts activated
            model.setObjective(gp.quicksum(expert_activation_indicators.values()), gp.GRB.MINIMIZE)
            # constraint: sum of each row of x_i,j = 0 if x_i,0 = 0 else k_experts_per_token
            for i in range(len(expert_selection_per_node)):
                model.addConstr(x[i].sum() == k_experts_per_token * x[i][0])
            # constraint: sum of x_i <= max_batch_sizes
            model.addConstr(x.sum() == max_batch_size * k_experts_per_token)
            model.optimize()
            # get expert choices
            expert_choices = defaultdict(int)
            for i, xi in enumerate(x):
                for j, xij in enumerate(xi):
                    if xij.X > 0:
                        expert_choices[expert_selection_per_node[i][j]] += 1
    activated_experts = len(expert_choices)
    return activated_experts

def generate_random_expert_indices(n_samples: int, k_experts_each: int, n_experts: int):
    expert_selection_per_node = []
    for _ in range(n_samples):
        expert_selection_per_node.append(tuple(random.sample(range(n_experts), k_experts_each)))
    return expert_selection_per_node

def calculate_distribution(n_samples: int,
                           k_experts_each: int,
                           k_candidate_experts: int,
                           n_experts: int,
                           max_batch_size: int,
                           n_trials: int = 1000):
    counts = defaultdict(int)
    for _ in tqdm.trange(n_trials):
        expert_selection_per_node = generate_random_expert_indices(n_samples, k_candidate_experts, n_experts)
        activated_experts = find_best_tokens_and_experts(expert_selection_per_node,
                                                         k_experts_per_token=k_experts_each,
                                                         max_batch_size=max_batch_size)
        counts[activated_experts] += 1
    return {k: v / n_trials for k, v in counts.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--k_experts_each", type=int, default=2)
    parser.add_argument("--k_candidate_experts", type=int, default=2)
    parser.add_argument("--max_batch_size", required=True, type=int)
    args = parser.parse_args()
    n_samples = args.n_samples
    n_experts = args.n_experts
    k_experts_each = args.k_experts_each
    k_candidate_experts = args.k_candidate_experts
    max_batch_size = args.max_batch_size
    distribution = calculate_distribution(n_samples, k_experts_each, k_candidate_experts, n_experts, max_batch_size)
    path = (f"distribution_ne{n_experts}_k{k_experts_each}"
           f"{'_relaxtop' + str(args.k_candidate_experts) if not args.k_candidate_experts == args.k_experts_each else ''}"
            ".csv")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("Activated Experts,Probability,Samples,Batch Size\n")
    with open(path, "a") as f:
        for k, v in distribution.items():
            f.write("{},{},{},{}\n".format(k, v, n_samples, max_batch_size))

if __name__ == "__main__":
    main()
