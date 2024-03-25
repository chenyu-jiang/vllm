import argparse
import os

import numpy as np
import torch

from vllm.model_executor.weight_utils import hf_model_weights_iterator

from filelock import FileLock

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

    def prune_neurons(self, neurons_to_maintain):
        self.w1.weight.data = self.w1.weight.data[neurons_to_maintain, :]
        self.w2.weight.data = self.w2.weight.data[:, neurons_to_maintain]
        self.w3.weight.data = self.w3.weight.data[neurons_to_maintain, :]

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
    # load neuron importance
    activation_norm = torch.load(os.path.join(args.output_dir, get_output_subdir(args), "activation_norm.pt"))
    # get top neurons
    top_neurons = torch.argsort(activation_norm, descending=True)[:int(activation_norm.shape[0] * args.top_neuron_percent)]
    # create model
    model = create_model(args, weights)
    model.eval()

    acticvation = model(data)
    # now remove top neurons
    model.prune_neurons(top_neurons)
    zeroed_out_activation = model(data)
    # calculate norm diff
    norm_diff = torch.linalg.norm(acticvation - zeroed_out_activation, dim=1).mean()

    print(f"Norm diff: {norm_diff.item()}")
    if args.output_to_csv:
        lock = FileLock(args.output_to_csv + ".lock")
        with lock:
            if not os.path.exists(args.output_to_csv):
                with open(args.output_to_csv, "w") as f:
                    f.write("expert_id,layer_id,top_neuron_percent,norm_diff\n")
            with open(args.output_to_csv, "a") as f:
                f.write(f"{args.expert_id},{args.layer_id},{args.top_neuron_percent},{norm_diff.item()}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--top_neuron_percent", type=float, default=0.1)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_to_csv", type=str, default="./neuron_importance_eval.csv")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, get_output_subdir(args))):
        os.makedirs(os.path.join(args.output_dir, get_output_subdir(args)), exist_ok=True)

    main_func(args)


if __name__ == "__main__":
    main()
