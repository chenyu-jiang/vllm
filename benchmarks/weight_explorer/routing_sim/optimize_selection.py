from typing import List

import os
import argparse
import random

import gurobipy as gp

from tqdm import trange

def indicator_constraint(binary_z, linexpr, M):
    # models the constraint linexpr >= 0 -> z = 1
    return linexpr - M * binary_z <= 0

def evaluate_expert_selection(expert_per_layer: List[List[List[int]]], experts_evicted_per_layer: List[List[int]]):
    n_tokens = len(expert_per_layer)
    n_layers = len(expert_per_layer[0])

    tokens_rerouted_count = [0 for _ in range(n_tokens)]
    for token_idx in range(n_tokens):
        for layer_idx in range(n_layers):
            for this_layer_expert in expert_per_layer[token_idx][layer_idx]:
                if this_layer_expert in experts_evicted_per_layer[layer_idx]:
                    tokens_rerouted_count[token_idx] += 1
    return tokens_rerouted_count

def solve_opt_selection(expert_per_layer: List[List[List[int]]], k_expert_to_evict: int, objective_type: str, max_experts=8):
    # expert_per_layer: [token, layer, experts]
    m = gp.Model()
    n_layers = len(expert_per_layer[0])
    n_tokens = len(expert_per_layer)
    evicted_experts = m.addMVar((n_layers, max_experts), vtype=gp.GRB.BINARY, name="evicted_experts")
    # create auxiliary variables
    if objective_type == "indicator":
        z = m.addMVar(n_tokens, vtype=gp.GRB.BINARY, name="z")
        for token_idx, layer_expert in enumerate(expert_per_layer):
            linexpr = gp.LinExpr()
            for layer_idx, experts in enumerate(layer_expert):
                for expert in experts:
                    linexpr += evicted_experts[layer_idx, expert]
            m.addConstr(indicator_constraint(z[token_idx], linexpr, (k_expert_to_evict + 1) * n_layers))
    elif objective_type == "sum":
        z = m.addMVar(n_tokens, name="z")
        for token_idx, layer_expert in enumerate(expert_per_layer):
            linexpr = gp.LinExpr()
            for layer_idx, experts in enumerate(layer_expert):
                for expert in experts:
                    linexpr += evicted_experts[layer_idx, expert]
            m.addConstr(z[token_idx] == linexpr)
    m.addConstr(evicted_experts.sum(axis=1) == k_expert_to_evict)
    m.setObjective(z.sum(), gp.GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.optimize()
    return m.ObjVal, evicted_experts.X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--n_layers", type=int, default=32)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--k_expert_to_evict", type=int, default=2)
    parser.add_argument("--k_expert_to_select", type=int, default=2)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--objective", type=str, default="indicator", choices=["indicator", "sum"])
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if args.output_path is None:
        args.output_path = f"./results_{args.objective}.csv"

    for batch_size in args.batch_sizes:
        obj_vals = []
        reroute_counts = {}
        for _ in trange(args.n_repeats):
            # generate random expert selection
            expert_per_layer = []
            for token in range(batch_size):
                expert_per_layer.append([])
                for layer in range(args.n_layers):
                    expert_per_layer[token].append(random.sample(range(args.n_experts), args.k_expert_to_select))
            # solve optimization problem
            obj_val, optimal_expert_selection = solve_opt_selection(expert_per_layer, args.k_expert_to_evict, args.objective, args.n_experts)
            optimal_expert_selection_list = []
            for layer_idx in range(args.n_layers):
                layer_list = []
                for expert_idx in range(args.n_experts):
                    if optimal_expert_selection[layer_idx, expert_idx] > 0:
                        layer_list.append(expert_idx)
                optimal_expert_selection_list.append(layer_list)
            # evaluate
            rerouted_count = evaluate_expert_selection(expert_per_layer, optimal_expert_selection_list)
            for reroute_count in rerouted_count:
                reroute_count = int(reroute_count)
                if reroute_count not in reroute_counts:
                    reroute_counts[reroute_count] = 0
                reroute_counts[reroute_count] += 1
            obj_vals.append(obj_val)
        obj_val = sum(obj_vals) / len(obj_vals)
        for k, v in reroute_counts.items():
            reroute_counts[k] = v / args.n_repeats
        if args.output_path:
            if not os.path.exists(args.output_path):
                with open(args.output_path, "w") as f:
                    if args.objective == "indicator":
                        f.write("batch_size,tokens_intact\n")
                    elif args.objective == "sum":
                        f.write("batch_size,routed_count,token_count\n")
            with open(args.output_path, "a") as f:
                if args.objective == "indicator":
                    f.write(f"{batch_size},{batch_size - int(obj_val)}\n")
                elif args.objective == "sum":
                    for k, v in reroute_counts.items():
                        f.write(f"{batch_size},{k},{v}\n")

if __name__ == "__main__":
    main()