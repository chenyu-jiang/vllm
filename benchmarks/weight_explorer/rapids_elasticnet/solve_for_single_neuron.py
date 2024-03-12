import argparse
import os
import filelock
import glob
from typing import List, Optional, Iterator, Tuple

import numpy as np
import torch
import time
import tqdm
import json

import numba
from numpy.typing import NDArray

from huggingface_hub import snapshot_download

from safetensors.torch import safe_open

# import snapml
import glmnet

# import cuml
# from cuml.common.device_selection import using_device_type, set_global_device_type

import multiprocessing as mp
import queue

global_other_neurons = None

def _to_torch_dtype(dtype):
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype, None)
        if dtype is None:
            raise ValueError(f"Invalid dtype {dtype}")
    return dtype

def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock

class Disabledtqdm(tqdm.auto.tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm,
                                          revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors

def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        load_format=load_format,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)

    if load_format == "npcache":
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()

@numba.njit
def concatenate_numba(arrays: List[np.ndarray]):
    n = len(arrays)
    m = arrays[0].shape[0]
    k = arrays[0].shape[1]
    res = np.zeros((m, n * k), dtype=arrays[0].dtype)
    for i in range(n):
        res[:, i * k:(i + 1) * k] = arrays[i]
    return res

# def solve_problem_glmnet(candidate: NDArray, target: NDArray, n_lambda=10, alpha: float = 0.9):
def solve_problem_glmnet(target: NDArray, n_lambda=10, alpha: float = 0.9):
    candidate_cpu = global_other_neurons
    target_cpu = target
    model = glmnet.ElasticNet(alpha=alpha, fit_intercept=False, n_jobs=1, n_splits=0, n_lambda=n_lambda, min_lambda_ratio=1e-6, standardize=False)
    # t = time.time()
    model.fit(candidate_cpu, target_cpu)
    # print("Time to fit model: ", time.time() - t)
    lambdas = model.lambda_path_
    coeffs = model.coef_path_
    return lambdas, coeffs

def mp_process(rcv_queue: mp.Queue, snd_queue: mp.Queue):
    while True:
        try:
            args = rcv_queue.get()
            if args is None:
                break
            neuron_id = args[0]
            lambdas, coeffs = solve_problem_glmnet(args[1])
            snd_queue.put((neuron_id, lambdas, coeffs))
        except Exception as e:
            snd_queue.put(e)

# def solve_problem_cuml(candidate: torch.Tensor, target: torch.Tensor, lambda_path: List[float] = None, n_lambda=50, alpha: float = 0.9):
#     candidate_numba = numba.cuda.as_cuda_array(candidate)
#     target_numba = numba.cuda.as_cuda_array(target)
#     if lambda_path is None:
#         lambda_path = list(np.logspace(np.log10(1e-6), np.log10(1e-1), n_lambda))
#     coeffs = []
#     with using_device_type("GPU"):
#         for lamb in tqdm.tqdm(lambda_path, desc="lambda"):
#             t = time.time()
#             # model = cuml.ElasticNet(alpha=lamb, l1_ratio=alpha, fit_intercept=False, output_type="numba")
#             model = cuml.Lasso(alpha=lamb, fit_intercept=False, output_type="numba", selection="random")
#             print("Time to create model: ", time.time() - t)
#             t = time.time()
#             model.fit(candidate_numba, target_numba)
#             print("Time to fit model: ", time.time() - t)
#             # get weights
#             coeffs.append(model.coef_)
#             import code
#             code.interact(local=locals())
#     return lambda_path, concatenate_numba(coeffs)

# def solve_problem_snapml(candidate: torch.Tensor, target: torch.Tensor, lambda_path: List[float] = None, n_lambda=50, alpha: float = 0.9):
#     candidate_numba = numba.cuda.as_cuda_array(candidate)
#     target_numba = numba.cuda.as_cuda_array(target)
#     if lambda_path is None:
#         lambda_path = list(np.logspace(np.log10(1e-6), np.log10(1e-1), n_lambda))
#     coeffs = []
#     for lamb in tqdm.tqdm(lambda_path, desc="lambda"):
#         t = time.time()
#         model = cuml.Lasso(alpha=lamb, fit_intercept=False, output_type="numba", selection="random")
#         print("Time to create model: ", time.time() - t)
#         t = time.time()
#         model.fit(candidate_numba, target_numba)
#         print("Time to fit model: ", time.time() - t)
#         # get weights
#         coeffs.append(model.coef_)
#         import code
#         code.interact(local=locals())
#     return lambda_path, concatenate_numba(coeffs)

def get_output_subdir(args):
    return f"l{args.layer_id}_e{args.expert_id}_w{args.weight_id}" + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")

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
    weights : List[torch.Tensor] = []
    for expert_id in range(8):
        weights.append(exper_id_to_params[expert_id])

    output_dir = os.path.join(args.out_dir, get_output_subdir(args))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # eval_f = open(os.path.join(output_dir, "eval.csv"), "w")
    # eval_f.write("neuron_id,lambda,nonzero_entries,norm_diff,element_wise_diff\n")
    n_proc = 96
    send_queue = mp.Queue()
    recv_queue = mp.Queue()
    neuron_weights = {}
    current_neuron_id = 0
    processes = []
    requests_sent = 0
    requests_received = 0

    max_neuron_id = weights[0].shape[0]
    # max_neuron_id = 3
    pbar = tqdm.tqdm(total=max_neuron_id, desc="Layer {} Expert {} Weight {}".format(args.layer_id, args.expert_id, args.weight_id), position=0, leave=True, unit="neuron")

    def _process_return_vals():
        nonlocal requests_received
        while True:
            try:
                ret = recv_queue.get_nowait()
                if isinstance(ret, Exception):
                    raise ret
                neuron_id, lambdas, coeffs = ret
                requests_received += 1
                neuron_weights[neuron_id] = (torch.tensor(lambdas), torch.from_numpy(coeffs))
                pbar.update(1)
            except queue.Empty:
                break
        time.sleep(0.5)

    other_neurons = torch.cat([w for i, w in enumerate(weights) if i != args.expert_id], dim=0)
    global global_other_neurons
    global_other_neurons = other_neurons.t().float().numpy()
    del other_neurons

    # launch workers
    for i in range(n_proc):
        p = mp.Process(target=mp_process, args=(send_queue, recv_queue))
        p.start()
        processes.append(p)

    while current_neuron_id < max_neuron_id:
        while requests_sent - requests_received < n_proc and current_neuron_id < max_neuron_id:
            target_neuron: torch.Tensor = weights[args.expert_id][current_neuron_id, :]
            # get all the neurons of other experts
            send_queue.put((current_neuron_id, target_neuron.float().numpy()))
            current_neuron_id += 1
            requests_sent += 1
        # try to wait for some workers to finish
        _process_return_vals()

    # wait for all workers to finish
    while requests_sent - requests_received > 0:
        _process_return_vals()

    # send stop signal
    for i in range(n_proc):
        send_queue.put(None)
    # join all processes
    for p in processes:
        p.join()
    print("Finished processing {} neurons.".format(len(neuron_weights)))

    # for neuron_id in range(weights[0].shape[0]):
    # for neuron_id in [0]:
        # # get the corresponding neuron
        # target_neuron: torch.Tensor = weights[args.expert_id][neuron_id, :]

        # # get all the neurons of other experts
        # other_neurons = torch.cat([w for i, w in enumerate(weights) if i != args.expert_id], dim=0)

        # # lambdas, coeffs = solve_problem_cuml(other_neurons.t().float(), target_neuron.float(), alpha=0.9)
        # # lambdas, coeffs = solve_problem_glmnet(other_neurons.t().float(), target_neuron.float(), alpha=0.9)
        # lambdas_torch = torch.tensor(lambdas)
        # # coeffs_torch = torch.from_numpy(coeffs.copy_to_host())
        # coeffs_torch = torch.from_numpy(coeffs)
        # neuron_weights[neuron_id] = (lambdas_torch, coeffs_torch)
        # # run eval
        # coeffs_torch_gpu = coeffs_torch.to(args.device)
        # other_neurons_double = other_neurons.to(coeffs_torch_gpu.dtype)
        # target_neuron_double = target_neuron.to(coeffs_torch_gpu.dtype)
        # for lamb, coeff in zip(lambdas, coeffs_torch_gpu.T):
        #     nonzeros = torch.sum(torch.abs(coeff) > 1e-3).item()
        #     # reconstruct using coeff
        #     reconstructed = other_neurons_double.t().to(coeff.dtype) @ coeff
        #     diff = target_neuron_double - reconstructed
        #     diff_norm = torch.norm(diff).item()
        #     diff_element_wise = torch.mean(torch.abs(diff)).item()
        #     # write to file
        #     eval_f.write(f"{neuron_id},{lamb},{nonzeros},{diff_norm},{diff_element_wise}\n")
        # eval_f.flush()
    # eval_f.close()
    # save the weights
    torch.save(neuron_weights, os.path.join(output_dir, "neuron_weights.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--weight_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    args = parser.parse_args()

    assert args.device in ["cuda", "cpu"], "Device must be in {'cuda', 'cpu'}"
    assert args.dtype in ["bfloat16", "float32"], "Dtype must be in {'bfloat16', 'float32'}"
    assert args.expert_id >= 0 and args.expert_id <= 7, "Expert ID must be in {0, 1, ..., 7}"
    assert args.weight_id in [1,2,3] , "Weight ID must be in {1, 2, 3}"
    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    main_func(args)


if __name__ == "__main__":
    main()
