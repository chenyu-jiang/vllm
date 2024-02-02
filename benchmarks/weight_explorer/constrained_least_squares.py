import gurobipy as gp
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from numpy.typing import NDArray
import tqdm
import glmnet
import torch

from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

def solve_problem_l1_penalized(target: NDArray, candidate: NDArray, l1_lambda: float):
    projected_dim = candidate.shape[1]
    with gp.Env(empty=True) as env:
        # env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(projected_dim, name="x", lb=-gp.GRB.INFINITY)
            norm_1 = model.addVar(name="norm")
            # least squares loss
            candidate_sq = candidate.T @ candidate
            target_candidate = target.T @ candidate
            target_target = target.T @ target

            model.setObjective(x.T @ candidate_sq @ x - 2 * target_candidate @ x + target_target + l1_lambda * norm_1, gp.GRB.MINIMIZE)
            # cardinality constraint
            model.addGenConstrNorm(norm_1, x, which=1, name="norm_constr")
            # optimize
            print("Optimizing...")
            model.optimize()
            return x.X

def solve_problem_sklearn_lasso(target: NDArray, candidate: NDArray, l1_lambda: float):
    # least squares loss
    model = Lasso(alpha=l1_lambda, fit_intercept=False, selection="random", tol=1e-6, max_iter=10000)
    model.fit(candidate, target)
    return model.coef_

def solve_problem_sklearn_linear(target: NDArray, candidate: NDArray):
    # least squares loss
    model = LinearRegression(fit_intercept=False)
    model.fit(candidate, target)
    return model.coef_

def solve_problem_glmnet(target: NDArray, candidate: NDArray):
    glmnet_lasso = glmnet.ElasticNet(alpha=1.0, fit_intercept=False, n_jobs=32, n_splits=0)
    glmnet_lasso.fit(candidate, target)
    lambdas = glmnet_lasso.lambda_path_
    coeffs = glmnet_lasso.coef_path_
    return lambdas, coeffs

def find_most_similar_column(target: NDArray, candidate: NDArray):
    best_norm = np.inf
    best_col = None
    for col_id in range(candidate.shape[1]):
        norm = np.linalg.norm(target - candidate[:, col_id])
        if norm < best_norm:
            best_norm = norm
            best_col = col_id
    return best_col

def main():
    # generate random data
    k = 64

    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1"):
        if "layers.0" in name and "experts" in name and "w1" in name:
            expert_id = int(name.split(".")[5])
            exper_id_to_params[expert_id] = param.float().numpy()

    f = open(f"./expert_reconstruct_all_ks_avg_experts_down_proj.csv", "w")
    f.write("expert_id,col_id,k,n_candidate_exps,l1_lambda,mean_diff,max_diff,norm\n")
    f.flush()

    def _predict(c, cand, k1):
        # select top k coefficients and zero out the rest
        ind = torch.argsort(torch.abs(c))[-k1:]
        new_c = torch.zeros_like(c)
        new_c[ind] = c[ind]
        # ind = np.argsort(np.abs(c))[-k1:]
        # new_c = np.zeros_like(c)
        # new_c[ind] = c[ind]
        return cand @ new_c.float()

    candidate_ks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for target_expert in tqdm.trange(8, desc="Target expert"):
        for col_id in [512]:
        # for col_id in tqdm.trange(0, 14336, 512, desc="Column", leave=False):
            target = exper_id_to_params[target_expert][:, col_id]
            target_cuda = torch.tensor(target).cuda()
            # candidate = np.concatenate([exper_id_to_params[i] for i in range(8) if i != target_expert], axis=1)
            # candidate = np.concatenate([exper_id_to_params[0], exper_id_to_params[1]], axis=1)
            candidate = (torch.sum(torch.stack([torch.tensor(exper_id_to_params[i]).cuda() for i in range(8)]), dim=0) / 8).squeeze().cpu().numpy()
            candidate_cuda = torch.tensor(candidate).cuda()
            # normalizer
            # normalizer = np.linalg.norm(target)
            # normalizer = 0.01
            # print("Normalizer: ", normalizer)
            # normed_target = target / normalizer
            # normed_candidate = candidate / normalizer
            # find the best L1 lambda
            best_norm = {k: np.inf for k in candidate_ks}
            best_mean_diff = {k: None for k in candidate_ks}
            best_max_diff = {k: None for k in candidate_ks}
            best_l1_lambda = {k: None for k in candidate_ks}
            lambdas, coeffs = solve_problem_glmnet(target, candidate)
            for l1_lambda, coeff in zip(lambdas, coeffs.T):
            # for l1_lambda in tqdm.tqdm([10, 5, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6], leave=False, desc="L1 lambda"):
                # coeffs = solve_problem_l1_penalized(target, candidate, l1_lambda)
                # coeffs = solve_problem_sklearn_lasso(normed_target, normed_candidate, l1_lambda)
                # coeffs = solve_problem_glmnet(target, candidate)
                # coeffs = solve_problem_sklearn_linear(target, candidate)
                nonzeros = np.count_nonzero(coeff)
                tqdm.tqdm.write("L1 lambda: {}, nonzero entries in solution: {}".format(l1_lambda, nonzeros))
                if (nonzeros == 0):
                    continue
                coeff_cuda = torch.tensor(coeff).cuda()
                for k in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                    # predicted = _predict(coeff, candidate, k)
                    predicted = _predict(coeff_cuda, candidate_cuda, k)
                    # col_id = find_most_similar_column(target, candidate)
                    # predicted = candidate[:, col_id]
                    # mean_diff = np.mean(np.abs(target - predicted))
                    # max_diff = np.max(np.abs(target - predicted))
                    # norm = np.linalg.norm(target - predicted)
                    mean_diff = torch.mean(torch.abs(target_cuda - predicted)).item()
                    max_diff = torch.max(torch.abs(target_cuda - predicted)).item()
                    norm = torch.norm(target_cuda - predicted).item()
                    if norm < best_norm[k]:
                        best_norm[k] = norm
                        best_mean_diff[k] = mean_diff
                        best_max_diff[k] = max_diff
                        best_l1_lambda[k] = l1_lambda
            for k in candidate_ks:
                f.write(f"{target_expert},{col_id},{k},7,{best_l1_lambda[k]},{best_mean_diff[k]},{best_max_diff[k]},{best_norm[k]}\n")
                f.flush()
    f.close()

if __name__ == "__main__":
    main()


