import argparse
import os

import tqdm

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

def evaluate(model: torch.nn.Module, eval_loader):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch, label in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            inputs = batch
            outputs = model(inputs)
            all_outputs.extend(list(outputs.float().cpu().numpy()))
            all_labels.extend(list(label.float().cpu().numpy()))
    # calculate spearman correlation
    from scipy.stats import spearmanr
    corr_spearman, _ = spearmanr(all_outputs, all_labels)
    # calculate pearson correlation
    from scipy.stats import pearsonr
    corr_pearson, _ = pearsonr(all_outputs, all_labels)
    return corr_spearman, corr_pearson

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
    label = label.to(args.device).to(_to_torch_dtype(args.dtype))
    train_size = int(args.train_split * len(data))
    train_data = data[:train_size]
    train_label = label[:train_size]
    eval_data = data[train_size:]
    eval_label = label[train_size:]
    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
    train_loader = DataLoader(TwoTensorDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TwoTensorDataset(eval_data, eval_label), batch_size=args.batch_size, shuffle=False)
    return train_loader, eval_loader

def main_func(args):
    # load data
    _, eval_loader = create_dataloaders(args)
    # create model
    model = Predictor(4096, args.r).to(args.device).to(_to_torch_dtype(args.dtype))
    # load weights
    weight_path = os.path.join(args.output_dir, f"e{args.expert_id}_l{args.layer_id}", "model.pt")
    assert os.path.exists(weight_path), f"Weight path {weight_path} does not exist"
    model.load_state_dict(torch.load(weight_path))
    corr_spearman, corr_pearson = evaluate(model, eval_loader)
    # write to csv file
    if not os.path.exists(os.path.join(args.output_dir, "correlation.csv")):
        with open(os.path.join(args.output_dir, "correlation.csv"), "w") as f:
            f.write("expert_id,layer_id,corr_spearman,corr_pearson\n")
    with open(os.path.join(args.output_dir, "correlation.csv"), "a") as f:
        f.write(f"{args.expert_id},{args.layer_id},{corr_spearman},{corr_pearson}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--expert_id", type=int, default=0)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    assert args.layer_id >= 0 and args.layer_id <= 31, "Layer ID must be in {0, 1, ..., 31}"

    fix_seed(42)
    main_func(args)


if __name__ == "__main__":
    main()
