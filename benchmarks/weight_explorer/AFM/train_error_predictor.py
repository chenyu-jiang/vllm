import argparse
import os

import tqdm

import wandb

import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader

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

class Predictor(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super(Predictor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, 1)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

def get_output_subdir(args):
    return f"e{args.expert_id}_l{args.layer_id}"

def evaluate(model: torch.nn.Module, eval_loader):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for batch, label in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            inputs = batch
            outputs = model(inputs)
            loss = F.mse_loss(outputs, label)
            avg_loss += torch.mean(loss).item()
    model.train()
    return avg_loss / len(eval_loader)

def train(args, model: torch.nn.Module, optimizer, train_loader, eval_loader, lr_scheduler=None, save=True):
    model.train()
    avg_loss = 0
    accum_iter = 0
    for epoch in range(args.num_epochs):
        for batch_idx, (batch, label) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            step_idx = epoch * len(train_loader) + batch_idx
            optimizer.zero_grad()
            outputs = model(batch)
            loss = F.mse_loss(outputs, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            accum_iter += 1
            # record loss
            if batch_idx % args.log_interval == 0:
                wandb.log({"train_loss": avg_loss / accum_iter,
                            "step": step_idx})
                avg_loss = 0
                accum_iter = 0
            if batch_idx % args.eval_interval == 0:
                eval_loss = evaluate(model, eval_loader)
                wandb.log({"eval_loss": eval_loss,
                            "step": step_idx})
            if lr_scheduler is not None:
                lr_scheduler.step()
    # final evaluation
    eval_loss = evaluate(model, eval_loader)
    wandb.log({"eval_loss": eval_loss,
                "step": step_idx})
    # save weights
    if save:
        save_dir = os.path.join(args.output_dir, get_output_subdir(args))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

class TwoTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    print("Loaded data with shape", data.shape, "dtype", data.dtype)
    # load label
    label = torch.load(os.path.join(args.label_dir, f"l{args.layer_id}_e{args.expert_id}.pt"))

    data = torch.tensor(data, dtype=_to_torch_dtype(args.dtype), device=args.device)
    label = torch.tensor(label, dtype=_to_torch_dtype(args.dtype), device=args.device)
    train_size = int(args.train_split * len(data))
    train_data = data[:train_size]
    train_label = label[:train_size]
    eval_data = data[train_size:]
    eval_label = label[train_size:]
    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
    train_loader = DataLoader(TwoTensorDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TwoTensorDataset(eval_data, eval_label), batch_size=args.batch_size, shuffle=False)
    return train_loader, eval_loader

def main_func(args, save=True):
    # load data
    train_loader, eval_loader = create_dataloaders(args)
    # create model
    model = Predictor(4096, args.r).to(args.device).to(_to_torch_dtype(args.dtype))
    # create optimizer
    optimizer = torch.optim.Adam([t for t in model.parameters() if t.requires_grad], lr=1e-3)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))
    # train
    train(args, model, optimizer, train_loader, eval_loader, lr_scheduler, save=save)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    wandb.init(project="error_predictor", dir=args.output_dir, config=args, name=get_output_subdir(args))
    main_func(args)
    wandb.finish()


if __name__ == "__main__":
    main()
