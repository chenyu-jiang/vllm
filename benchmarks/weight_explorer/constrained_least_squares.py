import gurobipy as gp
import numpy as np
from sklearn.linear_model import Lasso
from numpy.typing import NDArray
import tqdm

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
    model = Lasso(alpha=l1_lambda, fit_intercept=False, selection="random", tol=1e-4, max_iter=10000)
    model.fit(candidate, target)
    return model.coef_

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
        if "layers.0" in name and "experts" in name and "w2" in name:
            expert_id = int(name.split(".")[5])
            exper_id_to_params[expert_id] = param.float().numpy()

    f = open(f"./expert_reconstruct_all_ks.csv", "w")
    f.write("expert_id,col_id,k,n_candidate_exps,mean_diff,max_diff,norm\n")
    f.flush()

    def _predict(c, cand, k1):
        # select top k coefficients and zero out the rest
        ind = np.argsort(np.abs(c))[-k1:]
        import code
        code.interact(local=locals())
        new_c = np.zeros_like(c)
        new_c[ind] = c[ind]
        return cand @ new_c

    candidate_ks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for target_expert in tqdm.trange(8, desc="Target expert"):
        for col_id in [512]:
        # for col_id in tqdm.trange(0, 14336, 512, desc="Column", leave=False):
            target = exper_id_to_params[target_expert][:, col_id]
            candidate = np.concatenate([exper_id_to_params[i] for i in range(8) if i != target_expert], axis=1)
            # find the best L1 lambda
            best_norm = {k: np.inf for k in candidate_ks}
            best_mean_diff = {k: None for k in candidate_ks}
            best_max_diff = {k: None for k in candidate_ks}
            for l1_lambda in tqdm.tqdm([100, 10, 1, 0.1, 0.01, 0.001, 1e-4], leave=False, desc="L1 lambda"):
                # coeffs = solve_problem_l1_penalized(target, candidate, l1_lambda)
                coeffs = solve_problem_sklearn_lasso(target, candidate, l1_lambda)
                for k in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                    predicted = _predict(coeffs, candidate, k)
                    # col_id = find_most_similar_column(target, candidate)
                    # predicted = candidate[:, col_id]
                    mean_diff = np.mean(np.abs(target - predicted))
                    max_diff = np.max(np.abs(target - predicted))
                    norm = np.linalg.norm(target - predicted)
                    if norm < best_norm[k]:
                        best_norm[k] = norm
                        best_mean_diff[k] = mean_diff
                        best_max_diff[k] = max_diff
            for k in candidate_ks:
                f.write(f"{target_expert},{col_id},{k},7,{best_mean_diff[k]},{best_max_diff[k]},{best_norm[k]}\n")
                f.flush()
    f.close()

if __name__ == "__main__":
    main()


