import argparse
import os

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import tqdm

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

class MixtralExpertMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super(MixtralExpertMLP, self).__init__()
        self.w1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act_fn = torch.nn.SiLU()
        # self.freeze_weights()


    def freeze_weights(self):
        self.w1.requires_grad_(False)
        self.w2.requires_grad_(False)
        self.w3.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw_w1_out = self.w1(hidden_states)
        w1_out = self.act_fn(raw_w1_out)
        w3_out = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    def zero_out_neurons(self, neurons_to_zero_out):
        self.w1.weight.data[neurons_to_zero_out, :] = 0
        self.w2.weight.data[:, neurons_to_zero_out] = 0
        self.w3.weight.data[neurons_to_zero_out, :] = 0

    def reset_neurons(self, neurons_to_reset, w1, w2, w3):
        self.w1.weight.data[neurons_to_reset, :] = w1[neurons_to_reset, :]
        self.w2.weight.data[:, neurons_to_reset] = w2[:, neurons_to_reset]
        self.w3.weight.data[neurons_to_reset, :] = w3[neurons_to_reset, :]

    @classmethod
    def from_weights(cls, w1, w2, w3, dtype):
        hidden_dim = w1.shape[1]
        ffn_dim = w1.shape[0]
        model = cls(hidden_dim, ffn_dim)
        model = model.to(_to_torch_dtype(dtype))
        model.w1.weight.data.copy_(w1)
        model.w2.weight.data.copy_(w2)
        model.w3.weight.data.copy_(w3)
        # model.freeze_weights()
        return model

def create_model(args, weights):
    model = MixtralExpertMLP.from_weights(*weights, _to_torch_dtype(args.dtype))
    model = model.to(args.device)
    return model

def calc_norm_diff(zeroed_out_model, data, ref_out):
    with torch.no_grad():
        out_zerod = zeroed_out_model(data)
        norm_diff = torch.linalg.norm(ref_out - out_zerod)
    return norm_diff.item()

def calc_grad_wrt_norm(model, data):
    out = model(data)
    loss = torch.linalg.norm(out, dim=1).mean()
    loss.backward()
    # get grad of weitghts
    w1_grad = model.w1.weight.grad
    w2_grad = model.w2.weight.grad
    w3_grad = model.w3.weight.grad
    return w1_grad, w2_grad, w3_grad

def calc_grad_for_each_input(model, data):
    data.requires_grad_(True)
    out = model(data)
    loss = torch.linalg.norm(out, dim=1).mean()
    loss.backward()
    # get grad for d
    grad_d = data.grad
    return grad_d


def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
    return data

def get_output_subdir(args):
    return f"e{args.expert_id}_l{args.layer_id}"

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
    model.train()

    grad_over_dataset = calc_grad_wrt_norm(model, data)
    grad_inputs = calc_grad_for_each_input(model, data)
    torch.save(grad_over_dataset, os.path.join(args.output_dir, get_output_subdir(args), "weight_grad.pt"))
    torch.save(grad_inputs, os.path.join(args.output_dir, get_output_subdir(args), "input_grad.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, get_output_subdir(args))):
        os.makedirs(os.path.join(args.output_dir, get_output_subdir(args)), exist_ok=True)

    main_func(args)


if __name__ == "__main__":
    main()
