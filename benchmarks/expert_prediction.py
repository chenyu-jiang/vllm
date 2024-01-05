import os
import argparse
import random
from argparse import Namespace
import dataclasses
from dataclasses import dataclass
from collections import defaultdict

import wandb
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers import AutoTokenizer

from vllm.transformers_utils.config import get_config
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import hf_model_weights_iterator

def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to

class FrozenEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, params_dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_embeddings_padded = pad_vocab_size(num_embeddings)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.weight = Parameter(torch.empty(num_embeddings,
                                            embedding_dim,
                                            device=torch.cuda.current_device(),
                                            dtype=params_dtype))
        set_weight_attrs(self.weight, {
            "parallel_dim": 0,
            "weight_loader": self.weight_loader
        })
        self.weight.requires_grad = False

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, input_):
        return F.embedding(input_, self.weight)

class BOWRegressor(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_prediction_heads, regressor_hidden, n_experts, params_dtype=None):
        super().__init__()
        self.n_prediction_heads = n_prediction_heads
        self.embedding = FrozenEmbedding(num_embeddings, embedding_dim, params_dtype)
        self.fc = torch.nn.Linear(embedding_dim, regressor_hidden)
        for i in range(n_prediction_heads):
            setattr(self, f"output_{i}", torch.nn.Linear(regressor_hidden, n_experts))

    def forward(self, input_: torch.LongTensor):
        x = self.embedding(input_)
        # pool over sequence length
        x = x.sum(dim=1)
        x = self.fc(x)
        outputs = []
        for i in range(self.n_prediction_heads):
            outputs.append(F.softmax(getattr(self, f"output_{i}")(x), dim=-1))
        return outputs

@dataclass
class ExperimentConfig:
    seed: int = 42
    lr: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    architecture: str = "BOWRegressor"
    lr_scheduler: str = "CosineAnnealingLR"

    @classmethod
    def from_dict(cls, d):
        instance = cls()
        if isinstance(d, dict):
            d = d.items()
        elif isinstance(d, Namespace):
            d = vars(d).items()
        for k, v in d:
            if hasattr(instance, k):
                setattr(instance, k, v)
        return instance

class ExpertPredictionDataset:
    def __init__(self, data_dir, train_split=0.8, val_split=0.1):
        # parse test_dump_expert_ids.tsv
        expert_choices = defaultdict(dict) # token_id -> layer_id -> expert_ids
        # parse test_dump_expert_ids.tsv
        with open(os.path.join(data_dir, "test_dump_expert_ids.tsv"), "r") as f:
            f.readline() # skip header
            for line in f:
                token_id, layer_id, expert_ids = line.strip().split("\t")
                expert_ids = expert_ids.split(",")
                token_id = int(token_id)
                layer_id = int(layer_id)
                expert_ids = [int(expert_id) for expert_id in expert_ids]
                expert_choices[token_id][layer_id] = expert_ids
        # parse test_dump_token_ids.tsv
        token_contexts = {} # token_id -> context (list of token ids)
        with open(os.path.join(data_dir, "test_dump_token_ids.tsv"), "r") as f:
            f.readline() # skip header
            for line in f:
                token_id, context, _ = line.strip().split("\t")
                token_id = int(token_id)
                context = [int(token_id) for token_id in context.split(",")]
                token_contexts[token_id] = context
        # construct dataset of context -> expert_ids per layer
        dataset = []
        for token_id, context in token_contexts.items():
            if token_id not in expert_choices:
                continue
            per_layer_expert_ids = []
            n_layers = len(expert_choices[token_id])
            for layer_id in range(n_layers):
                expert_ids = expert_choices[token_id][layer_id]
                per_layer_expert_ids.append(expert_ids)
            dataset.append((context, per_layer_expert_ids))
        self.dataset = dataset
        # split dataset into train, val, test
        random.shuffle(self.dataset)
        n_train = int(len(self.dataset) * train_split)
        n_val = int(len(self.dataset) * val_split)
        self.train_dataset = self.dataset[:n_train]
        self.eval_dataset = self.dataset[n_train:n_train+n_val]
        self.test_dataset = self.dataset[n_train+n_val:]

def fix_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def create_collate_fn(model_config):
    def collate_fn(batch):
        max_len = max(len(context) for context, _ in batch)
        input_ = []
        labels = []
        for context, per_layer_expert_ids in batch:
            input_.append(torch.tensor(context + [0] * (max_len - len(context)), dtype=torch.long, device=torch.cuda.current_device()))
            stacked_labels = torch.zeros((model_config.num_hidden_layers, model_config.num_local_experts), dtype=torch.long, device=torch.cuda.current_device())
            for layer_id, expert_ids in enumerate(per_layer_expert_ids):
                stacked_labels[layer_id][expert_ids] = 1
            labels.append(stacked_labels)
        input_ = torch.stack(input_) # (batch_size, max_len)
        labels = torch.stack(labels) # (batch_size, num_hidden_layers, num_local_experts)
        return input_, labels
    return collate_fn


def init_model(exp_config: ExperimentConfig):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config = get_config(model_name, False)
    if exp_config.architecture == "BOWRegressor":
        model = BOWRegressor(config.vocab_size, config.hidden_size, config.num_hidden_layers, 4 * config.hidden_size, config.num_local_experts)
    else:
        raise NotImplementedError(f"Unknown architecture: {exp_config.architecture}")
    for name, loaded_weight in hf_model_weights_iterator(
                    model_name,
                    fall_back_to_pt=False):
        if name == "model.embed_tokens.weight":
            model.embedding.weight_loader(model.embedding.weight, loaded_weight)
            break
    return config, model

def init_wandb(config: ExperimentConfig):
    wandb.init(
        # set the wandb project where this run will be logged
        project="expert-prediction",
        # track hyperparameters and run metadata
        config=dataclasses.asdict(config),
    )

def train(args):
    config = ExperimentConfig.from_dict(args)
    init_wandb(config)
    fix_seed(config.seed)
    dataset = ExpertPredictionDataset(args.data_dir)
    print(f"Train dataset size: {len(dataset.train_dataset)}")
    print(f"Eval dataset size: {len(dataset.eval_dataset)}")
    print(f"Test dataset size: {len(dataset.test_dataset)}")
    model_config, model = init_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        raise NotImplementedError(f"Unknown lr scheduler: {config.lr_scheduler}")
    model = model.cuda()

    data_loader = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=create_collate_fn(model_config))
    for epoch in trange(config.epochs, desc="Epoch"):
        avg_loss = 0
        for i, (input_, labels) in tqdm(enumerate(data_loader), desc="Step", total=len(data_loader)):
            optimizer.zero_grad()
            outputs = model(input_)
            loss = 0
            per_output_head_inputs = [x.squeeze() for x in torch.split(labels, 1, dim=1)]
            for output, label in zip(outputs, per_output_head_inputs):
                loss += F.multilabel_soft_margin_loss(output, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if i != 0 and i % args.log_interval == 0:
                avg_loss /= args.log_interval
                wandb.log({"train_loss": avg_loss, "step": epoch * len(data_loader) + i, "epoch": epoch})
                avg_loss = 0
            if i != 0 and i % args.eval_interval == 0:
                eval_loss = 0
                total_samples = 0
                per_layer_prediction_matched = {layer_id: 0 for layer_id in range(model_config.num_hidden_layers)}
                per_layer_prediction_missed = {layer_id: 0 for layer_id in range(model_config.num_hidden_layers)}
                per_layer_prediction_extra = {layer_id: 0 for layer_id in range(model_config.num_hidden_layers)}
                with torch.no_grad():
                    eval_dataloader = torch.utils.data.DataLoader(dataset.eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=create_collate_fn(model_config))
                    for input_, labels in eval_dataloader:
                        outputs = model(input_)
                        loss = 0
                        per_output_head_inputs = [x.squeeze() for x in torch.split(labels, 1, dim=1)]
                        for layer_id, (output, label) in enumerate(zip(outputs, per_output_head_inputs)):
                            loss += F.multilabel_soft_margin_loss(output, label)
                            # sample the output logits
                            sampled_output = torch.topk(output, model_config.num_local_experts, dim=-1)[1].squeeze()
                            sampled_label = torch.topk(label, model_config.num_local_experts, dim=-1)[1].squeeze()
                            for output_expert_ids, label_expert_ids in zip(sampled_output, sampled_label):
                                output_expert_ids = output_expert_ids.tolist()
                                label_expert_ids = label_expert_ids.tolist()
                                n_matching = len(set(output_expert_ids) & set(label_expert_ids))
                                n_output_but_not_label = len(set(output_expert_ids) - set(label_expert_ids))
                                n_label_but_not_output = len(set(label_expert_ids) - set(output_expert_ids))
                                per_layer_prediction_matched[layer_id] += n_matching
                                per_layer_prediction_missed[layer_id] += n_label_but_not_output
                                per_layer_prediction_extra[layer_id] += n_output_but_not_label
                                total_samples += len(output_expert_ids)
                        eval_loss += loss.item()
                eval_loss /= len(data_loader)
                eval_per_layer_accuracy = {
                    layer_id: per_layer_prediction_matched[layer_id] / total_samples
                    for layer_id in range(model_config.num_hidden_layers)
                }
                eval_per_layer_precision = {
                    layer_id: per_layer_prediction_matched[layer_id] / (per_layer_prediction_matched[layer_id] + per_layer_prediction_extra[layer_id])
                    for layer_id in range(model_config.num_hidden_layers)
                }
                eval_per_layer_recall = {
                    layer_id: per_layer_prediction_matched[layer_id] / (per_layer_prediction_matched[layer_id] + per_layer_prediction_missed[layer_id])
                    for layer_id in range(model_config.num_hidden_layers)
                }
                eval_per_layer_f1 = {
                    layer_id: 2 * eval_per_layer_precision[layer_id] * eval_per_layer_recall[layer_id] / (eval_per_layer_precision[layer_id] + eval_per_layer_recall[layer_id])
                    for layer_id in range(model_config.num_hidden_layers)
                }
                wandb.log({"eval_loss": eval_loss, "step": epoch * len(data_loader) + i, "epoch": epoch,
                           **{f"eval_accuracy_l{layer_id}": eval_per_layer_accuracy[layer_id] for layer_id in range(model_config.num_hidden_layers)},
                           **{f"eval_precision_l{layer_id}": eval_per_layer_precision[layer_id] for layer_id in range(model_config.num_hidden_layers)},
                           **{f"eval_recall_l{layer_id}": eval_per_layer_recall[layer_id] for layer_id in range(model_config.num_hidden_layers)},
                           **{f"eval_f1_l{layer_id}": eval_per_layer_f1[layer_id] for layer_id in range(model_config.num_hidden_layers)},
                           })
        lr_scheduler.step()
        wandb.log({"lr": lr_scheduler.get_last_lr()[0], "epoch": epoch})
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_{epoch}.pt"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--architecture", type=str, default="BOWRegressor")
    parser.add_argument("--lr-scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)







