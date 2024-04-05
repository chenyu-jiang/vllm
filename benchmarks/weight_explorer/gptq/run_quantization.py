import argparse
import os

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import tqdm

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from gptq import GPTQ
from quant import Quantizer

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

class MixtralExpertMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super(MixtralExpertMLP, self).__init__()
        self.w1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, apply_w2: bool = True) -> torch.Tensor:
        raw_w1_out = self.w1(hidden_states)
        w1_out = self.act_fn(raw_w1_out)
        w3_out = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        if apply_w2:
            current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    @classmethod
    def from_weights(cls, w1, w2, w3, dtype):
        hidden_dim = w1.shape[1]
        ffn_dim = w1.shape[0]
        model = cls(hidden_dim, ffn_dim)
        model = model.to(_to_torch_dtype(dtype))
        model.w1.weight.data.copy_(w1)
        model.w2.weight.data.copy_(w2)
        model.w3.weight.data.copy_(w3)
        return model

def create_model(args, weights):
    model = MixtralExpertMLP.from_weights(*weights, _to_torch_dtype(args.dtype))
    model = model.to(args.device)
    return model

def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
    return data

def get_output_subdir(args):
    return f"e{args.expert_id}_l{args.layer_id}_w{args.wbits}"

def main_func(args):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1", fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    weights = [exper_id_to_params[args.expert_id][f"w{i}"] for i in range(1, 4)]
    # load data
    data = create_dataloaders(args)
    # create model
    model = create_model(args, weights)
    model.eval()

    quantized_model = create_model(args, weights)
    quantized_model.eval()

    # first quantize w1 and w3
    w1_gptq = GPTQ(quantized_model.w1)
    w1_quantizer = Quantizer()
    w1_quantizer.configure(bits=args.wbits, perchannel=True, sym=args.sym, mse=False)
    w1_gptq.quantizer = w1_quantizer
    w1_gptq.add_batch(data)
    w1_gptq.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups)

    w3_gptq = GPTQ(quantized_model.w3)
    w3_quantizer = Quantizer()
    w3_quantizer.configure(bits=args.wbits, perchannel=True, sym=args.sym, mse=False)
    w3_gptq.quantizer = w3_quantizer
    w3_gptq.add_batch(data)
    w3_gptq.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups)

    # then quantize w2
    w2_data = model(data, apply_w2=False)
    w2_gptq = GPTQ(quantized_model.w2)
    w2_quantizer = Quantizer()
    w2_quantizer.configure(bits=args.wbits, perchannel=True, sym=args.sym, mse=False)
    w2_gptq.quantizer = w2_quantizer
    w2_gptq.add_batch(w2_data)
    w2_gptq.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups)

    # run evaluation
    with torch.no_grad():
        model_output = model(data)
        quantized_model_output = quantized_model(data)
        diff_norm = torch.mean(torch.norm(model_output - quantized_model_output, dim=-1))
        print(f"Mean diff norm: {diff_norm}")
    
    with open(os.path.join(args.output_dir, get_output_subdir(args), "diff_norm.txt"), "w") as f:
        f.write(f"{diff_norm.item()}\n")

    # save quantized model
    torch.save(quantized_model.state_dict(), os.path.join(args.output_dir, get_output_subdir(args), "model.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--static_groups", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, get_output_subdir(args))):
        os.makedirs(os.path.join(args.output_dir, get_output_subdir(args)), exist_ok=True)

    main_func(args)


if __name__ == "__main__":
    main()
