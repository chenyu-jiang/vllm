import argparse
import os
from typing import List

import pytorch_lightning as pl

import gurobipy as gp

import numpy as np
import torch
import time
import tqdm

import glmnet

from torch.utils.data import DataLoader, TensorDataset
from numpy.typing import NDArray

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

def evaluate(merged_weight: torch.Tensor, weights: List[torch.Tensor], xs: List[torch.Tensor]):
    per_expert_diff_norm_mean = []
    per_expert_diff_norm_dist = []
    per_expert_element_wise_diff = []
    for x, w in zip(xs, weights):
        diff_mat = x @ (merged_weight - w.t())
        diff = torch.norm(diff_mat, dim=1)
        element_wise_diff = torch.mean(torch.abs(diff_mat))
        per_expert_diff_norm_mean.append(torch.mean(diff).item())
        per_expert_diff_norm_dist.append(list(diff.float().cpu().numpy()))
        per_expert_element_wise_diff.append(element_wise_diff.item())
    return per_expert_diff_norm_mean, per_expert_diff_norm_dist, per_expert_element_wise_diff

def solve_problem_glmnet(candidate: NDArray, target: NDArray, alpha=1.0):
    glmnet_lasso = glmnet.ElasticNet(alpha=alpha, fit_intercept=False, n_jobs=64, min_lambda_ratio=1e-6, n_splits=0)
    glmnet_lasso.fit(candidate, target)
    lambdas = glmnet_lasso.lambda_path_
    coeffs = glmnet_lasso.coef_path_
    return lambdas, coeffs

def solve_problem_glmnet_relaxed(candidate: NDArray, target: NDArray, k: int = 2048, alpha=1.0, weighted=True):
    glmnet_lasso = glmnet.ElasticNet(alpha=alpha, fit_intercept=False, n_jobs=64, min_lambda_ratio=1e-6, n_splits=0, standardize=False)
    t = time.time()
    if weighted:
        glmnet_lasso.fit(candidate, target, sample_weight=target**2)
    else:
        glmnet_lasso.fit(candidate, target)
    print("Time to fit 1st Lasso: ", time.time() - t)
    lambdas = glmnet_lasso.lambda_path_
    coeffs = glmnet_lasso.coef_path_
    final_coeffs = []
    for i in tqdm.trange(len(lambdas)):
        current_coeffs = coeffs[:,i]
        # run a second time with the zero lambda
        non_zero_map = np.abs(current_coeffs) > 1e-3
        if np.sum(non_zero_map) == 0:
            final_coeffs.append(current_coeffs)
            continue
        reduced_candidate = candidate[:, non_zero_map]
        t = time.time()
        lasso1 = glmnet.ElasticNet(alpha=0.0, fit_intercept=False, n_jobs=64, lambda_path=[0], n_splits=0)
        if weighted:
            lasso1.fit(reduced_candidate, target, sample_weight=target**2)
        else:
            lasso1.fit(reduced_candidate, target)
        tqdm.tqdm.write("Time to fit 2nd Lasso: {}".format(time.time() - t))
        # restore the original shape
        final_coeff = np.zeros(candidate.shape[1]) 
        final_coeff[non_zero_map] = lasso1.coef_path_.squeeze()
        final_coeffs.append(final_coeff)
    final_coeffs = np.stack(final_coeffs, axis=1)
    return lambdas, coeffs, final_coeffs

def solve_problem_glmnet_relaxed_once(candidate: NDArray, target: NDArray, k: int = 2048, alpha=1.0, weighted=True):
    glmnet_lasso = glmnet.ElasticNet(alpha=alpha, fit_intercept=False, n_jobs=64, min_lambda_ratio=1e-6, n_splits=0, standardize=False)
    t = time.time()
    if weighted:
        glmnet_lasso.fit(candidate, target, sample_weight=target**2)
    else:
        glmnet_lasso.fit(candidate, target)
    lambdas = glmnet_lasso.lambda_path_
    coeffs = glmnet_lasso.coef_path_
    final_coeffs = []
    for i in range(len(lambdas)):
        current_coeffs = coeffs[:,i]
        # run a second time with the zero lambda
        non_zero_map = np.abs(current_coeffs) > 1e-3
        if np.sum(non_zero_map) >= k:
            # truncate to k
            non_zero_indices = np.argsort(np.abs(current_coeffs))[-k:]
            non_zero_map = np.zeros_like(non_zero_map)
            non_zero_map[non_zero_indices] = 1
            reduced_candidate = candidate[:, non_zero_map]
            lasso1 = glmnet.ElasticNet(alpha=0.0, fit_intercept=False, n_jobs=64, lambda_path=[0], n_splits=0)
            if weighted:
                lasso1.fit(reduced_candidate, target, sample_weight=target**2)
            else:
                lasso1.fit(reduced_candidate, target)
            # restore the original shape
            final_coeff = np.zeros(candidate.shape[1]) 
            final_coeff[non_zero_map] = lasso1.coef_path_.squeeze()
            print("Truncated to k: ", np.sum(np.abs(final_coeff) > 1e-3))
            lambdas = [lambdas[i]]
            coeffs = np.expand_dims(coeffs[:,i], axis=1)
            final_coeffs.append(final_coeff)
            break
    if not final_coeffs:
        lambdas = [lambdas[-1]]
        coeffs = np.expand_dims(coeffs[:,-1], axis=1)
        final_coeffs = np.expand_dims(coeffs[:,-1], axis=1)
    else:
        final_coeffs = np.stack(final_coeffs, axis=1)
    import code
    code.interact(local=locals())
    return lambdas, coeffs, final_coeffs


def get_output_subdir(args):
    return f"w{args.weight_id}_l{args.layer_id}" + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")

def least_squares(A: torch.Tensor, y: torch.Tensor):
    return torch.linalg.lstsq(A, y).solution

def sparse_least_squares(A: torch.Tensor, y: torch.Tensor, k: int):
    A_np = A.cpu().numpy()
    y_np = y.cpu().numpy()
    with gp.Env(empty=True) as env:
        # env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(A_np.shape[1], name="x")
            model.setObjective((A_np @ x - y_np) @ (A_np @ x - y_np), gp.GRB.MINIMIZE)
            # add constraint on nonzero entries
            z = model.addMVar(A_np.shape[1], vtype=gp.GRB.BINARY, name="z")
            model.addConstr(x <= 1000 * z)
            model.addConstr(z.sum() <= k)
            model.optimize()
            return x.X

def explore_right_inverse(A: torch.Tensor):
    A_inv = torch.linalg.lstsq(A.float(), torch.eye(A.shape[0]).to(A.device).float()).solution
    return A_inv

def explore_similarity(A: torch.tensor, y: torch.Tensor):
    # A: n x m
    # y: n
    # sort the columns of A by cosine distance with y
    # return the indices of the sorted columns
    A_norm = A / torch.norm(A, dim=0)
    y_norm = y / torch.norm(y)
    similarity = A_norm.t() @ y_norm
    indices = torch.argsort(similarity, descending=True)
    for i in range(10):
        print(similarity[indices[i]].item())


def plot_convergence(train_loss):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(train_loss)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    fig.savefig('train_loss.pdf', bbox_inches='tight')


def main_func(args):
    # load weights
    t = time.time()
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                 fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if f"w{args.weight_id}" in name:
                exper_id_to_params[expert_id] = param
    weights = []
    for expert_id in range(8):
        weights.append(exper_id_to_params[expert_id].to(args.device).reshape(-1, exper_id_to_params[expert_id].shape[-1] * 2))
    # get the first neuron of expert 0
    first_neuron = weights[0][0, :]
    print("Time to load weights: ", time.time() - t)
    print("Weights shape: ", weights[0].shape)

    # get all the neurons of expert 1,...,7
    t = time.time()
    other_neurons = torch.cat(weights[1:], dim=0)
    print("Time to concatenate weights: ", time.time() - t)

    # inv = explore_right_inverse(other_neurons.t())
    # adapter = inv.bfloat16() @ weights[0].t()

    # # get random tensor
    # random_tensor = torch.randn_like(first_neuron)
    # err = torch.norm(random_tensor @ weights[0].t() - random_tensor @ other_neurons.t() @ adapter)
    # print("Error: ", err.item())
    # import code
    # code.interact(local=locals())
    # exit(0)

    # explore_similarity(other_neurons.t(), first_neuron)
    # exit(0)

    # lambdas, coeffs = solve_problem_glmnet(other_neurons.t().cpu().float().numpy(), first_neuron.cpu().float().numpy(), alpha=0.9)
    lambdas, coeffs, final_coeffs = solve_problem_glmnet_relaxed_once(other_neurons.t().cpu().float().numpy(), first_neuron.cpu().float().numpy(), alpha=0.9, weighted=True, k=4096)
    for l1_lambda, coeff, f_coeff in zip(lambdas, coeffs.T, final_coeffs.T):
        nonzeros = np.sum(np.abs(coeff) > 1e-3)
        nonzero_final_coeffs = np.sum(np.abs(f_coeff) > 1e-3)
        
        # reconstruct using coeff
        # coeff_torch = torch.tensor(coeff, device=other_neurons.device, dtype=other_neurons.dtype)
        coeff_torch_large = torch.tensor(coeff, device=other_neurons.device, dtype=other_neurons.dtype)
        coeff_torch_large[coeff_torch_large.abs() < 1e-3] = 0
        # reconstructed = other_neurons.t() @ coeff_torch
        # diff = torch.norm(first_neuron - reconstructed)
        first_neuron_top_10percent_mask = torch.abs(first_neuron) > torch.quantile(torch.abs(first_neuron).float(), 0.9)
        first_neuron_top_5percent_mask = torch.abs(first_neuron) > torch.quantile(torch.abs(first_neuron).float(), 0.95)

        recon_large = other_neurons.t() @ coeff_torch_large
        diff_large = torch.norm(first_neuron - recon_large)
        diff_large_top10p = torch.norm(first_neuron[first_neuron_top_10percent_mask] - recon_large[first_neuron_top_10percent_mask])
        diff_large_top5p = torch.norm(first_neuron[first_neuron_top_5percent_mask] - recon_large[first_neuron_top_5percent_mask])

        final_coeff_torch_large = torch.tensor(f_coeff, device=other_neurons.device, dtype=other_neurons.dtype)
        final_coeff_torch_large[final_coeff_torch_large.abs() < 1e-3] = 0
        recon_final_large = other_neurons.t() @ final_coeff_torch_large
        diff_final_large = torch.norm(first_neuron - recon_final_large)
        diff_final_large_top10p = torch.norm(first_neuron[first_neuron_top_10percent_mask] - recon_final_large[first_neuron_top_10percent_mask])
        diff_final_large_top5p = torch.norm(first_neuron[first_neuron_top_5percent_mask] - recon_final_large[first_neuron_top_5percent_mask])
        print("Lambda: {}, nonzero entries in solution: {}, norm diff large: {}, top10p: {}, top5p: {}, final nonzero: {}, final norm diff large: {}, top10p: {}, top5p: {}".format(l1_lambda, nonzeros, diff_large.item(), diff_large_top10p.item(), diff_large_top5p.item(), nonzero_final_coeffs.item(), diff_final_large.item(), diff_final_large_top10p.item(), diff_final_large_top5p.item()))

    # weights = sparse_least_squares(other_neurons.t().float(), first_neuron.float(), 4096)
    # print(first_neuron.shape)
    # print(other_neurons.shape)
    # solution = least_squares(other_neurons.t().float(), first_neuron.float())
    # print(solution.shape)
    # model = ElasticLinear(torch.nn.MSELoss(), n_inputs=other_neurons.shape[0], l1_lambda=10, l2_lambda=10).to(args.device).to(_to_torch_dtype(args.dtype))
    # trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=1)
    # x = other_neurons.t()
    # y = first_neuron
    # loader = DataLoader(TensorDataset(x, y), batch_size=4096, shuffle=True)
    # print("Num of batches: ", len(loader))
    # trainer.fit(model, loader)
    # calculate the number of non-zero weights
    # print("Non zero entries: ", torch.sum(torch.abs(model.output_layer.weight) < 1e-4), " out of ", model.output_layer.weight.numel())
    # plot_convergence(model.train_log)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    args = parser.parse_args()

    assert args.weight_id in [1,2,3] , "Weight ID must be in {1, 2, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
