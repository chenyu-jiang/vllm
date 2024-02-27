import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

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

class FFN(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int):
        super(FFN, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, r)
        self.fc2 = torch.nn.Linear(r, out_dim)
        self.act = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class NNApprox(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int, r: int):
        super(NNApprox, self).__init__()
        self.k = k
        for i in range(k + 1):
            setattr(self, f"wk_1_{i}", torch.nn.Linear(in_dim, out_dim, bias=False))
            setattr(self, f"wk_3_{i}", torch.nn.Linear(in_dim, out_dim, bias=False))
            setattr(self, f"wk_2_{i}", torch.nn.Linear(out_dim, in_dim, bias=False))
        self.out_nn = FFN(in_dim * k, in_dim, r)
        self.act = torch.nn.SiLU()
        self.freeze_weights()

    def freeze_weights(self):
        for i in range(self.k + 1):
            getattr(self, f"wk_1_{i}").requires_grad_(False)
            getattr(self, f"wk_3_{i}").requires_grad_(False)
            getattr(self, f"wk_2_{i}").requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        wk_outs = []
        for i in range(self.k + 1):
            wk_1 = getattr(self, f"wk_1_{i}")
            wk_2 = getattr(self, f"wk_2_{i}")
            wk_3 = getattr(self, f"wk_3_{i}")
            wk_1_out = wk_1(hidden_states)
            wk_1_out = self.act(wk_1_out)
            wk_3_out = wk_3(hidden_states)
            wk_out = wk_1_out * wk_3_out
            wk_out = wk_2(wk_out)
            wk_outs.append(wk_out)
        concated_nn_outs = torch.cat(wk_outs[1:], dim=1)
        projected_outs = self.out_nn(concated_nn_outs)
        diff = wk_outs[0] - projected_outs
        return torch.linalg.vector_norm(diff, ord=2, dim=1), torch.mean(torch.abs(diff), dim=1)

    @classmethod
    def from_weights(cls, wks_1: List[torch.Tensor],
                     wks_3: List[torch.Tensor],
                     wks_2: List[torch.Tensor],
                     r: int, dtype):
        assert len(wks_1) > 1
        assert len(wks_3) > 1
        assert len(wks_2) > 1
        in_dim = wks_1[0].shape[1]
        out_dim = wks_1[0].shape[0]
        model = cls(in_dim, out_dim, len(wks_1) - 1, r)
        model = model.to(_to_torch_dtype(dtype))
        for i, w in enumerate(wks_1):
            wk_module = getattr(model, f"wk_1_{i}")
            wk_module.weight.data.copy_(w)
        for i, w in enumerate(wks_3):
            wk_module = getattr(model, f"wk_3_{i}")
            wk_module.weight.data.copy_(w)
        for i, w in enumerate(wks_2):
            wk_module = getattr(model, f"wk_2_{i}")
            wk_module.weight.data.copy_(w)
        model.freeze_weights()
        return model

    def get_adapter_states(self):
        state_dict = self.state_dict()
        return {k: v for k, v in state_dict.items() if "adapter" in k}

def create_model(args, weights_1, weights_3, weights_2, r):
    model = NNApprox.from_weights(weights_1, weights_3, weights_2, r, args.dtype)
    model = model.to(args.device)
    return model

def evaluate(model: torch.nn.Module, eval_loader):
    model.eval()
    avg_loss = 0
    avg_diff = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            inputs = batch
            outputs, mean_diff = model(inputs)
            avg_loss += torch.mean(outputs).item()
            avg_diff += torch.mean(mean_diff).item()
    model.train()
    return avg_loss / len(eval_loader), avg_diff / len(eval_loader)

def get_output_subdir(args):
    if len(args.target_expert_ids) == 7:
        expert_id_str = "all"
    elif len(args.target_expert_ids) == 0:
        expert_id_str = "rnd"
    else:
        expert_id_str = ",".join(map(str, args.target_expert_ids))
    return f"w{args.weight_id}_e{args.expert_id}_l{args.layer_id}_r{args.r}_e{expert_id_str}" + ("" if args.dtype == "bfloat16" else f"_{args.dtype}")

def train(args, model: NNApprox, train_loader, eval_loader, optimizer: torch.optim.Optimizer, lr_scheduler=None, save=True):
    model.train()
    avg_loss = 0
    avg_diff = 0
    accum_iter = 0
    for epoch in range(args.num_epochs):
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_idx = epoch * len(train_loader) + batch_idx
            optimizer.zero_grad()
            outputs, mean_diff = model(batch)
            loss = torch.mean(outputs)
            diff = torch.mean(mean_diff)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_diff += diff.item()
            accum_iter += 1
            # record loss
            if batch_idx % args.log_interval == 0:
                wandb.log({"train_loss": avg_loss / accum_iter,
                            "train_mean_diff": avg_diff / accum_iter,
                            "step": step_idx})
                avg_loss = 0
                avg_diff = 0
                accum_iter = 0
            if batch_idx % args.eval_interval == 0:
                eval_loss, eval_diff = evaluate(model, eval_loader)
                wandb.log({"eval_loss": eval_loss,
                            "eval_mean_diff": eval_diff,
                            "step": step_idx})
            if lr_scheduler is not None:
                lr_scheduler.step()
    # final evaluation
    eval_loss, eval_diff = evaluate(model, eval_loader)
    wandb.log({"eval_loss": eval_loss,
                "eval_mean_diff": eval_diff,
                "step": step_idx})
    # save weights
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.get_adapter_states(), os.path.join(save_dir, "model.pt"))


def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
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
    for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1",
                                                 fall_back_to_pt=False):
        if f"layers.{args.layer_id}" in name and "experts" in name:
            expert_id = int(name.split(".")[5])
            if expert_id not in exper_id_to_params:
                exper_id_to_params[expert_id] = {}
            for w_name in ["w1", "w2", "w3"]:
                if w_name in name:
                    exper_id_to_params[expert_id][w_name] = param
    target_weight_1 = exper_id_to_params[args.expert_id][f"w1"]
    target_weight_3 = exper_id_to_params[args.expert_id][f"w3"]
    target_weight_2 = exper_id_to_params[args.expert_id][f"w2"]
    if not args.target_expert_ids:
        assert False, "Not implemented"
    else:
        other_weights_1 = [exper_id_to_params[other_expert_id][f"w1"] for other_expert_id in args.target_expert_ids]
        other_weights_3 = [exper_id_to_params[other_expert_id][f"w3"] for other_expert_id in args.target_expert_ids]
        other_weights_2 = [exper_id_to_params[other_expert_id][f"w2"] for other_expert_id in args.target_expert_ids]
    # load data
    train_loader, eval_loader = create_dataloaders(args)
    # create model
    model = create_model(args, [target_weight_1] + other_weights_1,
                            [target_weight_3] + other_weights_3,
                            [target_weight_2] + other_weights_2,
                          args.r)
    # create optimizer
    optimizer = torch.optim.Adam([t for t in model.parameters() if t.requires_grad], lr=1e-3)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))
    # train
    train(args, model, train_loader, eval_loader, optimizer, lr_scheduler, save=save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--target_expert_ids", type=str, default="all")
    parser.add_argument("--weight_id", type=int, default=2)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.expert_id < 8 and args.expert_id >= 0, "Expert ID must be in [0, 7]"
    assert args.weight_id == 2, "Weight ID must be in [2]"

    if args.target_expert_ids == "all":
        args.target_expert_ids = sorted(list(set(range(8)) - {args.expert_id}))
    elif args.target_expert_ids == "rnd":
        args.target_expert_ids = []
    else:
        args.target_expert_ids = list(map(int, args.target_expert_ids.split(",")))

    fix_seed(42)
    wandb.init(project="multi-lowrank-all-others-nn", dir=args.output_dir, config=args, name=get_output_subdir(args))
    main_func(args)
    wandb.finish()


if __name__ == "__main__":
    main()
