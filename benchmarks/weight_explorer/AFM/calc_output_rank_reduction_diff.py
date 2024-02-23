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
        self.freeze_weights()


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
        return raw_w1_out, w3_out, current_hidden_states

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

def create_model(args, src_weights, dst_weights):
    src_model = MixtralExpertMLP.from_weights(*src_weights, _to_torch_dtype(args.dtype))
    dst_model = MixtralExpertMLP.from_weights(*dst_weights, _to_torch_dtype(args.dtype))
    src_model = src_model.to(args.device)
    dst_model = dst_model.to(args.device)
    return src_model, dst_model

def calc_norm_diff_after_rank_reduction(src_model, dst_model, eval_loader, ranks):
    src_model.eval()
    dst_model.eval()
    all_outputs = {
        "w1": [],
        "w3": [],
        "w2": [],
    }
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader)):
            inputs = batch
            w1_out, w3_out, w2_out = src_model(inputs)
            w1_out_dst, w3_out_dst, w2_out_dst = dst_model(inputs)
            all_outputs["w1"].append((w1_out_dst - w1_out).detach())
            all_outputs["w3"].append((w3_out_dst - w3_out).detach())
            all_outputs["w2"].append((w2_out_dst - w2_out).detach())
    norm_diffs = defaultdict(list)
    norm_diff_percents = defaultdict(list)
    mean_diffs = defaultdict(list)
    max_diffs = defaultdict(list)
    explained_vars = defaultdict(list)
    for w_name in tqdm.tqdm(["w1", "w3", "w2"], desc="Weight"):
        concat_output = torch.cat(all_outputs[w_name], dim=0)
        # somehow the svd fails if we use the entire dataset (65536)
        if concat_output.shape[1] > 4096:
            concat_output = concat_output[:16384, :]
        else:
            concat_output = concat_output[:32768, :]
        # calculate SVD
        U, S, Vh = torch.linalg.svd(concat_output)
        S_sq = S ** 2
        for rank in tqdm.tqdm(ranks, desc="Rank"):
            reduced_output = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            norm_diff = torch.linalg.norm(concat_output - reduced_output)
            norm_diff_percent = norm_diff / torch.linalg.norm(concat_output)
            mean_diff = torch.mean(torch.abs(concat_output - reduced_output))
            max_diff = torch.max(torch.abs(concat_output - reduced_output))
            explained_var = S_sq[:rank].sum() / S_sq.sum()
            norm_diffs[w_name].append(norm_diff.item())
            norm_diff_percents[w_name].append(norm_diff_percent.item())
            mean_diffs[w_name].append(mean_diff.item())
            max_diffs[w_name].append(max_diff.item())
            explained_vars[w_name].append(explained_var.item())
    return norm_diffs, norm_diff_percents, mean_diffs, max_diffs, explained_vars

def create_dataloaders(args):
    data = np.load(os.path.join(args.data_dir, f"expert_activations_e{args.dst_expert_id}_l{args.layer_id}_0.npz"))["arr_0"]
    data = data.reshape(-1, 4096)
    print("Loaded data with shape", data.shape, "dtype", data.dtype)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: default_collate([torch.tensor(x_, dtype=_to_torch_dtype(args.dtype), device=args.device) for x_ in x]))
    return data_loader

def main_func(args):
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
    src_weights = [exper_id_to_params[args.src_expert_id][f"w{i}"] for i in range(1, 4)]
    dst_weights = [exper_id_to_params[args.dst_expert_id][f"w{i}"] for i in range(1, 4)]
    # load data
    data_loader = create_dataloaders(args)
    # create model
    src_model, dst_model = create_model(args, src_weights, dst_weights)
    # calculate norm diff
    ranks = [8, 16, 32, 64, 128, 256, 512] + list(range(1024, 4096 + 512, 512))
    norm_diffs, norm_diff_percents, mean_diffs, max_diffs, explained_vars = calc_norm_diff_after_rank_reduction(src_model, dst_model, data_loader, ranks)
    # save results
    exists = os.path.exists(args.output_path)
    with open(args.output_path, "a") as f:
        if not exists:
            f.write("weight,src_expert_id,dst_expert_id,layer_id,rank,norm_diff,norm_lost_percent,mean_diff,max_diff,explained_var\n")
        for w_name in ["w1", "w3", "w2"]:
            for rank, norm_diff, norm_diff_percent, mean_diff, max_diff, explained_var in zip(ranks, norm_diffs[w_name], norm_diff_percents[w_name], mean_diffs[w_name], max_diffs[w_name], explained_vars[w_name]):
                f.write(f"{w_name},{args.src_expert_id},{args.dst_expert_id},{args.layer_id},{rank},{norm_diff},{1 - norm_diff_percent},{mean_diff},{max_diff},{explained_var}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--src_expert_id", type=int, default=0)
    parser.add_argument("--dst_expert_id", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    assert args.src_expert_id != args.dst_expert_id, "src_expert_id and dst_expert_id should be different"

    main_func(args)


if __name__ == "__main__":
    main()
