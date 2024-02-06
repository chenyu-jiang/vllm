import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import tqdm

import wandb

from peft import get_peft_model_state_dict, inject_adapter_in_model, LoraConfig

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
        self.freeze_weights()


    def freeze_weights(self):
        self.w1.requires_grad_(False)
        self.w2.requires_grad_(False)
        self.w3.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
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
        model.freeze_weights()
        return model

def create_model(args, from_weights, to_weights):
    config = LoraConfig(
        lora_alpha = args.lora_alpha,
        r=args.r,
        bias="none",
        target_modules=["w1", "w2", "w3"],
    )
    lora_model = MixtralExpertMLP.from_weights(*from_weights, _to_torch_dtype(args.dtype))
    lora_model = inject_adapter_in_model(config, lora_model)
    lora_model = lora_model.to(args.device)
    print("Using LoRA model", lora_model)
    ref_model = MixtralExpertMLP.from_weights(*to_weights, _to_torch_dtype(args.dtype))
    ref_model = ref_model.to(args.device)
    return lora_model, ref_model

def evaluate(lora_model, ref_model, eval_loader, loss_func):
    lora_model.eval()
    avg_loss = 0
    avg_mean_diff = 0
    avg_max_diff = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            inputs = batch
            lora_outputs = lora_model(inputs)
            ref_outputs = ref_model(inputs)
            loss = loss_func(lora_outputs, ref_outputs)
            avg_loss += loss.item()
            # mean diff
            mean_diff = torch.mean(torch.abs(lora_outputs - ref_outputs))
            # max diff
            max_diff = torch.max(torch.abs(lora_outputs - ref_outputs))
            avg_mean_diff += mean_diff.item()
            avg_max_diff += max_diff.item()
    lora_model.train()
    return avg_loss / len(eval_loader), avg_mean_diff / len(eval_loader), avg_max_diff / len(eval_loader)

def train(args, lora_model, ref_model, train_loader, eval_loader, loss_func, optimizer, lr_scheduler=None, save=True):
    lora_model.train()
    ref_model.eval()
    avg_loss = 0
    for epoch in range(args.num_epochs):
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_idx = epoch * len(train_loader) + batch_idx
            optimizer.zero_grad()
            lora_outputs = lora_model(batch)
            ref_outputs = ref_model(batch)
            loss = loss_func(lora_outputs, ref_outputs)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            # record loss
            if batch_idx % args.log_interval == 0:
                wandb.log({"train_loss": avg_loss / args.log_interval, "step": step_idx})
                avg_loss = 0
            if batch_idx % args.eval_interval == 0:
                eval_loss, mean_diff, max_diff = evaluate(lora_model, ref_model, eval_loader, loss_func)
                wandb.log({"eval_loss": eval_loss,
                            "mean_diff": mean_diff,
                            "max_diff": max_diff,
                            "step": step_idx})
            if lr_scheduler is not None:
                lr_scheduler.step()
    # final evaluation
    eval_loss, mean_diff, max_diff = evaluate(lora_model, ref_model, eval_loader, loss_func)
    wandb.log({"eval_loss": eval_loss,
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "step": step_idx})
    # save lora weights
    peft_state_dict = get_peft_model_state_dict(lora_model)
    if save:
        torch.save(peft_state_dict, os.path.join(args.output_dir, "lora_model.pt"))


def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.to_expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    print("Loaded data with shape", data.shape, "dtype", data.dtype)
    # shuffle data rows
    np.random.shuffle(data)
    train_size = int(args.train_split * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]
    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: default_collate([torch.tensor(x_, dtype=_to_torch_dtype(args.dtype), device=args.device) for x_ in x]))
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda x: default_collate([torch.tensor(x_, dtype=_to_torch_dtype(args.dtype), device=args.device) for x_ in x]))
    return train_loader, eval_loader

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
    from_weights = [exper_id_to_params[args.from_expert_id][f"w{i}"] for i in range(1, 4)]
    to_weights = [exper_id_to_params[args.to_expert_id][f"w{i}"] for i in range(1, 4)]
    # load data
    train_loader, eval_loader = create_dataloaders(args)
    # create model
    lora_model, ref_model = create_model(args, from_weights, to_weights)
    # create loss function
    loss_func = torch.nn.MSELoss()
    # create optimizer
    optimizer = torch.optim.Adam([t for t in lora_model.parameters() if t.requires_grad], lr=1e-3)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))
    # train
    train(args, lora_model, ref_model, train_loader, eval_loader, loss_func, optimizer, lr_scheduler, save=save)

def sweep_agent(config=None):
    with wandb.init(config=config):
        config = wandb.config
        main_func(config, save=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--from_expert_id", type=int, default=0)
    parser.add_argument("--to_expert_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    fix_seed(42)
    wandb.init(project="exlora", dir=args.output_dir, config=args, name=f"from{args.from_expert_id}_to{args.to_expert_id}_l{args.layer_id}_r{args.r}")
    main_func(args)
    wandb.finish()


if __name__ == "__main__":
    # main()
    sweep_config = {
        "name": "lora_transfer",
        "method": "grid",
        "metric": {
            "name": "eval_loss",
            "goal": "minimize",
        },
        "parameters": {
            "lora_alpha": {
                "values": [8, 16, 32, 64, 128, 256, 512]
            },
            "r": {
                "values": [8, 16, 32, 64, 128, 256, 512]
            },
            "device": { "value": "cuda" },
            "log_interval": { "value": 100 },
            "eval_interval": { "value": 2000 },
            "batch_size": { "value": 256 },
            "dtype": { "value": "bfloat16" },
            "num_epochs": { "value": 10 },
            "train_split": { "value": 0.9 },
            "from_expert_id": { "value": 0 },
            "to_expert_id": { "value": 1 },
            "layer_id": { "value": 31 },
            "data_dir": { "value": "/root/expert_data" },
            "output_dir": { "value": "/root/lora_exps/sweep" },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="exlora")
    wandb.agent(sweep_id, function=sweep_agent)
