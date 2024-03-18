import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn

from typing import List

import tqdm

import wandb

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

# copied from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer(torch.nn.Module):
    def __init__(
        self, 
        r: int, 
        in_features: int,
        out_features: int,
        lora_alpha: int, 
        lora_dropout: float = 0,
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.scaling = self.lora_alpha / self.r
        # Initialize lora B weights to zero
        with torch.no_grad():
            self.lora_B.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class MixtralExpertMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, n_loras: int, lora_r: int, freeze_weight: bool = False):
        super(MixtralExpertMLP, self).__init__()
        self.w1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act_fn = torch.nn.SiLU()
        self.n_loras = n_loras
        if n_loras > 0:
            self.w1_loras = torch.nn.ModuleList([LoRALayer(lora_r, hidden_dim, ffn_dim, lora_r) for _ in range(n_loras)])
            self.w3_loras = torch.nn.ModuleList([LoRALayer(lora_r, hidden_dim, ffn_dim, lora_r) for _ in range(n_loras)])
            self.w2_loras = torch.nn.ModuleList([LoRALayer(lora_r, ffn_dim, hidden_dim, lora_r) for _ in range(n_loras)])
        if freeze_weight:
            self.freeze_weights()

    def freeze_weights(self):
        self.w1.requires_grad_(False)
        self.w2.requires_grad_(False)
        self.w3.requires_grad_(False)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        if self.n_loras == 0:
            hidden_states = hidden_states.to(device=self.w1.weight.device, dtype=self.w1.weight.dtype)
            w1_out = self.w1(hidden_states)
            w1_out = self.act_fn(w1_out)
            w3_out = self.w3(hidden_states)
            current_hidden_states = w1_out * w3_out
            current_hidden_states = self.w2(current_hidden_states)
            return current_hidden_states
        assert len(hidden_states) == len(self.w1_loras), f"Expected {len(self.w1_loras)} hidden states, got {len(hidden_states)}"
        hidden_states = [hidden_states.to(device=self.w1.weight.device, dtype=self.w1.weight.dtype) for hidden_states in hidden_states]
        out_hidden_states = []
        for lora_idx, hs in enumerate(hidden_states):
            w1_out = self.w1(hs)
            w1_lora_out = self.w1_loras[lora_idx](hs)
            w1_out = self.act_fn(w1_out + w1_lora_out)
            w3_out = self.w3(hs)
            w3_lora_out = self.w3_loras[lora_idx](hs)
            w3_out = w3_out + w3_lora_out
            chs = w1_out * w3_out
            chs_w2 = self.w2(chs)
            chs_w2_lora = self.w2_loras[lora_idx](chs)
            chs = chs_w2 + chs_w2_lora
            out_hidden_states.append(chs)
        return out_hidden_states

    @classmethod
    def from_weights(cls, w1, w2, w3, dtype, n_loras, lora_r, freeze=False):
        hidden_dim = w1.shape[1]
        ffn_dim = w1.shape[0]
        model = cls(hidden_dim, ffn_dim, n_loras, lora_r, freeze)
        model = model.to(_to_torch_dtype(dtype))
        model.w1.weight.data.copy_(w1)
        model.w2.weight.data.copy_(w2)
        model.w3.weight.data.copy_(w3)
        return model

def evaluate(lora_model, ref_models, eval_loader, loss_func):
    lora_model.eval()
    avg_loss = 0
    avg_mean_diff = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            assert len(batch) == len(ref_models), f"Expected {len(ref_models)} inputs, got {len(batch)}"
            lora_outputs = lora_model(batch)
            ref_outputs = [ref_models[i](batch[i]) for i in range(len(ref_models))]
            ref_outputs = torch.stack(ref_outputs, dim=0)
            lora_outputs = torch.stack(lora_outputs, dim=0)
            loss = loss_func(lora_outputs, ref_outputs)
            avg_loss += loss.item()
            # mean diff
            mean_diff = torch.mean(torch.abs(lora_outputs - ref_outputs))
            avg_mean_diff += mean_diff.item()
    lora_model.train()
    return avg_loss / len(eval_loader), avg_mean_diff / len(eval_loader)

def train(args, lora_model, ref_models, train_loader, eval_loader, loss_func, optimizer, lr_scheduler=None, save=True):
    lora_model.train()
    for ref_model in ref_models:
        ref_model.eval()
    avg_loss = 0
    loss_counter = 0
    for epoch in range(args.num_epochs):
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_idx = epoch * len(train_loader) + batch_idx
            optimizer.zero_grad()
            lora_outputs = lora_model(batch)
            ref_outputs = [ref_models[i](batch[i]) for i in range(len(ref_models))]
            ref_outputs = torch.stack(ref_outputs, dim=0)
            lora_outputs = torch.stack(lora_outputs, dim=0)
            loss = loss_func(lora_outputs, ref_outputs)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            loss_counter += 1
            # record loss
            if batch_idx % args.log_interval == 0:
                wandb.log({"train_loss": avg_loss / loss_counter, "step": step_idx})
                avg_loss = 0
                loss_counter = 0
            if batch_idx % args.eval_interval == 0:
                eval_loss, mean_diff = evaluate(lora_model, ref_models, eval_loader, loss_func)
                wandb.log({"eval_loss": eval_loss,
                            "mean_diff": mean_diff,
                            "step": step_idx})
            if lr_scheduler is not None:
                lr_scheduler.step()
    # final evaluation
    eval_loss, mean_diff = evaluate(lora_model, ref_models, eval_loader, loss_func)
    wandb.log({"eval_loss": eval_loss,
                "mean_diff": mean_diff,
                "step": step_idx})
    if save:
        save_path = os.path.join(args.output_dir, get_output_subdir(args))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(lora_model.state_dict(), os.path.join(save_path, "lora_model.pt"))

def get_output_subdir(args):
    return f"l{args.layer_id}_r{args.r}" + ("_frozen" if args.freeze else "")

class DataIterator():
    def __init__(self, data_list: List[torch.Tensor], batch_size: int):
        self.data_list = data_list
        # check if all data have the same length
        self.length = len(data_list[0])
        for data in data_list:
            assert len(data) == self.length, "All data should have the same length"
        self.batch_size = batch_size
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.length:
            raise StopIteration
        batch = [data[self.current_idx:self.current_idx+self.batch_size] for data in self.data_list]
        self.current_idx += self.batch_size
        return batch

    def __len__(self):
        return self.length // self.batch_size

def create_dataloaders(args):
    expert_train_datasets = []
    expert_eval_datasets = []
    n_samples = None
    for expert_id in range(8):
        data = np.load(os.path.join(args.data_dir, f"expert_activations_e{expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
        data = data.reshape(-1, 4096)
        print("Loaded data for expert", expert_id, "with shape", data.shape, "dtype", data.dtype)
        np.random.shuffle(data)
        if n_samples is None:
            n_samples = len(data)
        else:
            assert n_samples == len(data), f"Expected {n_samples} samples, got {len(data)}"
        train_size = int(args.train_split * len(data))
        train_data = data[:train_size]
        eval_data = data[train_size:]
        expert_train_datasets.append(torch.tensor(train_data, dtype=_to_torch_dtype(args.dtype), device=args.device))
        expert_eval_datasets.append(torch.tensor(eval_data, dtype=_to_torch_dtype(args.dtype), device=args.device))
    train_iterator = DataIterator(expert_train_datasets, args.batch_size)
    eval_iterator = DataIterator(expert_eval_datasets, args.batch_size)
    return train_iterator, eval_iterator

def main_func(args, save=True):
    # load weights
    exper_id_to_params = {}
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1"):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    ref_models = []
    for expert_id in range(8):
        w1 = exper_id_to_params[expert_id]["w1"]
        w2 = exper_id_to_params[expert_id]["w2"]
        w3 = exper_id_to_params[expert_id]["w3"]
        model = MixtralExpertMLP.from_weights(w1, w2, w3, args.dtype, 0, 0)
        model.to(device=args.device, dtype=_to_torch_dtype(args.dtype))
        ref_models.append(model)
    # load from merged weights
    merged_w1 = torch.load(os.path.join(args.merged_dir, f"w1_l{args.layer_id}", "merged_weight.pt"))
    merged_w2 = torch.load(os.path.join(args.merged_dir, f"w2_l{args.layer_id}", "merged_weight.pt"))
    merged_w3 = torch.load(os.path.join(args.merged_dir, f"w3_l{args.layer_id}", "merged_weight.pt"))
    lora_model = MixtralExpertMLP.from_weights(merged_w1.t(), merged_w2.t(), merged_w3.t(), args.dtype, 8, args.r, freeze=args.freeze)
    lora_model.to(device=args.device, dtype=_to_torch_dtype(args.dtype))
    # load data
    train_loader, eval_loader = create_dataloaders(args)
    # create loss function
    loss_func = torch.nn.MSELoss()
    # create optimizer
    optimizer = torch.optim.Adam([t for t in lora_model.parameters() if t.requires_grad], lr=1e-3)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))
    # train
    train(args, lora_model, ref_models, train_loader, eval_loader, loss_func, optimizer, lr_scheduler, save=save)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    fix_seed(42)
    wandb.init(project="merged_lora", dir=args.output_dir, config=args, name=f"l{args.layer_id}_r{args.r}" + ("_frozen" if args.freeze else ""))
    main_func(args)
    wandb.finish()


if __name__ == "__main__":
    main()
