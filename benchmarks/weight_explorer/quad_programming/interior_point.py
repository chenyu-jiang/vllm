import torch
from torchmin import minimize as torch_minimize

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from typing import Callable

class SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        result = torch.log(inp)
        result[inp < 0] = float('-inf')
        ctx.save_for_backward(inp)
        return result
    
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_input = grad_output / inp
        grad_input[inp < 0] = float('inf')
        return grad_input

def interior_point(
    objective: Callable,
    inequality_constraint: Callable,
    x0: torch.Tensor,
    t_init: float = 1,
    mu: float = 0.5,
    tol: float = 1e-8,
    method: str = "l-bfgs",
):
    # implement interior point method
    # this uses logaritmic barrier
    # constaints: inequality_constraint(x) <= 0
    # x0: initial guess

    # assert x0 initial guess is feasible
    assert torch.all(inequality_constraint(x0) <= 0)

    def barrier(x, t):
        return -t * SafeLog.apply(-inequality_constraint(x))

    # t_current = t_init
    t_current = 0
    x_opt = x0
    iteration_count = 0
    while True:
        def augmented_objective(x):
            return torch.sum(objective(x)) # + barrier(x, t_current))

        # solve the unconstrained problem
        opt_result = torch_minimize(
            augmented_objective, x_opt, method=method, tol=tol
        )
        if not opt_result.success:
            raise RuntimeError("Optimization failed: " + opt_result.message)
        x_opt = opt_result.x

        # check stopping criterion
        if t_current < tol:
            break
        # update t
        t_current *= mu
        iteration_count += 1

    return x_opt


def ref_scipy(
    objective: Callable,
    inequality_constraint: Callable,
    x0: torch.Tensor,
    tol: float = 1e-5,
):
    # implement interior point method
    # this uses logaritmic barrier
    # constaints: inequality_constraint(x) <= 0
    #             it is assumed that inequality_constraint(x) returns a scalar
    # x0: initial guess

    scipy_constraint = NonlinearConstraint(
        inequality_constraint, - np.inf, 0
    )

    opt_result = minimize(objective, x0, constraints=[scipy_constraint], tol=tol)
    if not opt_result.success:
        raise RuntimeError("Optimization failed: " + opt_result.message)
    x_opt = opt_result.x

    return x_opt


def test_interior_point(dtype=torch.float, batched=False):
    # test the interior point method
    def objective(x):
        return torch.sum(x ** 2, dim=-1)

    def objective_np(x):
        return np.sum(x ** 2, axis=-1)

    def inequality_constraint(x):
        return 2 - torch.sum(x, dim=-1)

    def inequality_constraint_np(x):
        return 2 - np.sum(x, axis=-1)

    n_dim = 128
    if batched:
        n_dim = (8, 128)
    # make sure x0 is feasible
    x0_torch = torch.randn(n_dim, dtype=dtype).cuda() + 50
    x_opt_torch = interior_point(objective, inequality_constraint, x0_torch, tol=1e-4, method="l-bfgs")

    x0_np = x0_torch.cpu().numpy()
    if not batched:
        x_opt_np = ref_scipy(objective_np, inequality_constraint_np, x0_np)
    else:
        x_opt_np = []
        for i in range(n_dim[0]):
            x_opt_np_i = ref_scipy(objective_np, inequality_constraint_np, x0_np[i])
            x_opt_np.append(x_opt_np_i)
        x_opt_np = np.stack(x_opt_np)
    assert np.allclose(x_opt_torch.cpu().numpy(), x_opt_np, atol=1e-4)

if __name__ == "__main__":
    # test_interior_point()
    test_interior_point(batched=True)