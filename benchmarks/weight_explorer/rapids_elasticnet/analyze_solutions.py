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
        weights.append(exper_id_to_params[expert_id].to(args.device).float())

    output_dir = os.path.join(args.out_dir, get_output_subdir(args))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load neuron weights
    neuron_weights = torch.load(os.path.join(output_dir, "neuron_weights.pt"))

    other_neurons = torch.cat([w for i, w in enumerate(weights) if i != args.expert_id], dim=0).t().float().to(args.device)

    eval_f = open(os.path.join(output_dir, "eval.csv"), "w")
    eval_f.write("neuron_id,lambda,nonzeros,diff_norm,diff_element_wise\n")
    for current_neuron_id in tqdm.trange(weights[args.expert_id].shape[0]):
        target_neuron: torch.Tensor = weights[args.expert_id][current_neuron_id, :].float()
        lambdas, coeffs = neuron_weights[current_neuron_id]
        lambdas = lambdas.float()
        coeffs = coeffs.float()
        for lamb, coeff in zip(lambdas, coeffs.T):
            lamb = lamb.item()
            coeff = coeff.to(args.device)
            nonzeros = torch.sum(torch.abs(coeff) > 1e-3).item()
            # reconstruct using coeff
            reconstructed = other_neurons @ coeff
            diff = target_neuron - reconstructed
            diff_norm = torch.norm(diff).item()
            diff_element_wise = torch.mean(torch.abs(diff)).item()
            # write to file
            eval_f.write(f"{current_neuron_id},{lamb},{nonzeros},{diff_norm},{diff_element_wise}\n")
            eval_f.flush()

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
