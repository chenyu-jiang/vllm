import argparse

import torch
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.mixtral import MixtralMoE, MixtralParallelism
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel
)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

def load_weights_first_layer_dp(model, model_name):
    params_dict = dict(model.named_parameters())
    stacked_params_mapping = [
        ("w1_w3", "w1", 0),
        ("w1_w3", "w3", 1),
    ]
    for name, loaded_weight in hf_model_weights_iterator(model_name, fall_back_to_pt=False):
        if name.startswith("model.layers.1.block_sparse_moe."):
            name = name[len("model.layers.1.block_sparse_moe."):]
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ("experts." in name and name not in params_dict):
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

def load_weights_first_layer_tp(model, model_name):
    params_dict = dict(model.named_parameters())
    for name, loaded_weight in hf_model_weights_iterator(model_name, fall_back_to_pt=False):
        if name.startswith("model.layers.1.block_sparse_moe."):
            name = name[len("model.layers.1.block_sparse_moe."):]
            if ("experts." in name and name not in params_dict):
                    continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_benchmark(batch_size, parallel_method: MixtralParallelism) -> float:
    mixtal_config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", False)
    model = MixtralMoE(mixtal_config, parallel_method=parallel_method)
    model = model.cuda()
    model = model.to(torch.bfloat16)
    model.eval()
    rank = torch.distributed.get_rank()
    if parallel_method == MixtralParallelism.TENSOR_EXPERT_PARALLEL:
        load_weights_first_layer_tp(model, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    else:
        load_weights_first_layer_dp(model, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    set_seed(42)
    dummy_input = torch.randn(batch_size, 1, 4096).cuda().to(torch.bfloat16)
    if parallel_method == MixtralParallelism.DATA_EXPERT_PARALLEL:
        # split input across data parallel dimension
        assert batch_size % dp_size == 0
        dummy_input = dummy_input[rank * batch_size // dp_size: (rank + 1) * batch_size // dp_size]
        # dump input
        torch.save(dummy_input, f"./dp_dump_dummy_input_rank{rank}.pt")
    else:
        if rank == 0:
            # dump input
            torch.save(dummy_input, f"./tp_dump_dummy_input.pt")
    print("Dummy input shape: {}".format(dummy_input.shape))
    output = model(dummy_input)
    torch.cuda.synchronize()
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--parallel-method", type=str, choices=["TP_EP", "DP_TP_EP"], default="TP_EP")
    args = parser.parse_args()

    repeat = 20
    num_iters = 100
    warmup_iters = 20
    tp_size = args.tp_size
    dp_size = args.dp_size
    use_cuda_graph = args.use_cuda_graph
    if tp_size > 1 or dp_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=tp_size * dp_size,
        )
        initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                  data_parallel_size=dp_size)
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    print("Rank: {} initialized.".format(rank))
    parallel_method = MixtralParallelism.TENSOR_EXPERT_PARALLEL if args.parallel_method == "TP_EP" else MixtralParallelism.DATA_EXPERT_PARALLEL
    batch_size = 8 * dp_size * tp_size
    output = run_benchmark(batch_size, parallel_method)
    if parallel_method == MixtralParallelism.TENSOR_EXPERT_PARALLEL:
        if rank == 0:
            torch.save(output, f"./ep_output.pt")
    else:
        torch.save(output, f"./dp_output_rank{rank}.pt")