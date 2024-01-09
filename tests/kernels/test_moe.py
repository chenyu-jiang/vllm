import pytest
import random

import torch
import torch.nn.functional as F

from vllm._C import moe_ops


FP_DTYPES = [torch.half, torch.bfloat16, torch.float]
INT_DTYPES = [torch.int, torch.long]
NUM_TOKENS = [83]  # Arbitrary values for testing
NUM_EXPERTS = [2, 4, 8] 
SEEDS = [0]
HIDDEN_DIMS = [1024, 4096]
TOP_K = [2]
DEVICES = [i for i in range(1 if torch.cuda.device_count() == 1 else 2)]

def torch_ref(input_hidden_states, expert_indices, k, tokens, hidden_dim, num_experts, fp_dtype, int_dtype):
    capacity = tokens * k

    masks_se = torch.zeros((tokens, num_experts), dtype=int_dtype, device="cuda")
    for i in range(tokens):
        masks_se[i, expert_indices[i]] = 1

    locations_cumsum = torch.cumsum(masks_se, axis=0).to(int_dtype)
    locations1 = locations_cumsum - 1
    out_locations1 = torch.zeros((tokens, k), dtype=int_dtype, device="cuda")
    for i in range(tokens):
        for j in range(k):
            out_locations1[i, j] = locations1[i, expert_indices[i, j]]

    dispatched_input = torch.zeros((num_experts, capacity, hidden_dim), dtype=fp_dtype, device="cuda")
    for i in range(tokens):
        for j in range(k):
            dispatched_input[expert_indices[i, j], out_locations1[i, j], :] = input_hidden_states[i, :]

    return out_locations1, dispatched_input

def run_moe_gen_location(expert_indices, k, tokens, num_experts, dtype):
    output_locations = torch.zeros((tokens, k), dtype=dtype, device="cuda")
    moe_ops.moe_gen_location(output_locations, expert_indices, k, tokens, num_experts)
    return output_locations

def run_prepare_inputs(tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # set device
    # generate random input
    input_shape = (tokens, hidden_dim)
    input_hidden_states = torch.rand(*input_shape).cuda().to(dtype=fp_dtype)
    # generate random router outputs
    router_logits_shape = (tokens, num_experts)
    router_logits = torch.rand(*router_logits_shape, device="cuda", dtype=fp_dtype)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, expert_indices = torch.topk(routing_weights,
                                                    top_k,
                                                    dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(fp_dtype)
    expert_indices = expert_indices.to(int_dtype)
    return input_hidden_states, expert_indices, routing_weights

@pytest.mark.parametrize("tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp_dtype", FP_DTYPES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_moe_gen_location(tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype, device):
    torch.cuda.set_device(device)
    input_hidden_states, expert_indices, _ = run_prepare_inputs(
        tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype)

    torch_locations, _ = torch_ref(input_hidden_states, expert_indices, top_k,
                                   tokens, hidden_dim, num_experts, fp_dtype, int_dtype)
    kernel_locations = torch.zeros((tokens, top_k), dtype=int_dtype, device="cuda")
    moe_ops.moe_gen_location(kernel_locations, expert_indices, top_k, tokens, num_experts)
    assert torch.allclose(torch_locations, kernel_locations)

@pytest.mark.parametrize("tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp_dtype", FP_DTYPES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_moe_dispatch(tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype, device):
    torch.cuda.set_device(device)
    input_hidden_states, expert_indices, _ = run_prepare_inputs(
        tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype)

    torch_locations, torch_dispatched = torch_ref(input_hidden_states, expert_indices, top_k,
                                   tokens, hidden_dim, num_experts, fp_dtype, int_dtype)

    kernel_dispatched = torch.zeros((num_experts, tokens * top_k, hidden_dim), dtype=fp_dtype, device="cuda")
    moe_ops.moe_dispatch(kernel_dispatched, expert_indices, torch_locations, input_hidden_states, top_k, tokens, hidden_dim, num_experts)
    assert torch.allclose(torch_dispatched, kernel_dispatched)

@pytest.mark.parametrize("tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp_dtype", FP_DTYPES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_moe_gather(tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype, device):
    torch.cuda.set_device(device)
    input_hidden_states, expert_indices, routing_weights = run_prepare_inputs(
        tokens, hidden_dim, num_experts, top_k, seed, fp_dtype, int_dtype)

    torch_locations, torch_dispatched = torch_ref(input_hidden_states,
                                                  expert_indices,
                                                  top_k,
                                                  tokens,
                                                  hidden_dim,
                                                  num_experts,
                                                  fp_dtype,
                                                  int_dtype)
    # here we assume experts performed identity function
    # so gathered output = dispatched input
    torch_gathered = input_hidden_states
    kernel_gathered = torch.zeros((tokens, hidden_dim), dtype=fp_dtype, device="cuda")
    moe_ops.moe_gather(kernel_gathered,
                       torch_dispatched,
                       routing_weights,
                       expert_indices,
                       torch_locations,
                       top_k,
                       tokens,
                       hidden_dim)
    # reduce checking precision for half and bfloat16 since we are not
    # actually performing computation in torch_ref
    if fp_dtype == torch.half:
        assert torch.allclose(torch_gathered, kernel_gathered, atol=1e-3)
    elif fp_dtype == torch.bfloat16:
        assert torch.allclose(torch_gathered, kernel_gathered, atol=1e-2)
    else:
        assert torch.allclose(torch_gathered, kernel_gathered)


if __name__ == "__main__":
    pytest.main([__file__])