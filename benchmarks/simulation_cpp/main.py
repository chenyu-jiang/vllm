import os
import pickle
from collections import defaultdict
from typing import List, Tuple, Optional, Callable
import argparse
import tqdm
import math
import time

import simulator

from vllm.transformers_utils.cost_model import ProfileBasedCostModel


def load_dataset(
    dataset_dir: str,
    max_samples: int,
    max_decoded_tokens: int = None,
    max_layers: int = None,
    repeat_tokens: int = 1,
    repeat: int = 1,
):
    # TODO: hard code file names for now
    # parse test_dump_expert_ids.tsv
    n_layers = 0
    n_experts = 0
    k_experts_per_token = None
    # token_id -> layer_id -> expert_ids
    token_id_to_experts = defaultdict(lambda: defaultdict(int))

    if not os.path.exists(os.path.join(dataset_dir, "dataset.pkl")):
        expert_ids_fn = os.path.join(dataset_dir, "test_dump_expert_ids.tsv")
        with tqdm.tqdm(
            total=os.path.getsize(expert_ids_fn), desc="Parsing expert ids"
        ) as pbar:
            with open(expert_ids_fn, "r") as f:
                l = f.readline()  # skip header
                pbar.update(len(l))
                for line in f:
                    token_id, layer_id, expert_ids = line.strip().split("\t")
                    expert_ids = expert_ids.split(",")
                    token_id = int(token_id)
                    layer_id = int(layer_id)
                    n_layers = max(n_layers, layer_id + 1)
                    n_experts = max(
                        n_experts,
                        max([int(expert_id) for expert_id in expert_ids]) + 1,
                    )
                    expert_ids = [int(expert_id) for expert_id in expert_ids]
                    token_id_to_experts[token_id][layer_id] = sorted(
                        expert_ids
                    )
                    pbar.update(len(line))
        # parse test_dump_token_ids.tsv
        token_id_to_contexts = []
        token_id_to_output_token = {}
        token_ids_fn = os.path.join(dataset_dir, "test_dump_token_ids.tsv")
        with tqdm.tqdm(
            total=os.path.getsize(token_ids_fn), desc="Parsing token ids"
        ) as pbar:
            with open(token_ids_fn, "r") as f:
                l = f.readline()  # skip header
                pbar.update(len(l))
                for line in f:
                    token_id, context, output_token = line.strip().split("\t")
                    token_id = int(token_id)
                    context = [int(token) for token in context.split(",")]
                    output_token = int(output_token)
                    token_id_to_contexts.append((token_id, context))
                    token_id_to_output_token[token_id] = output_token
                    pbar.update(len(line))
        # organize tokens into requests
        unique_sequences: List[Tuple[Tuple[int], Tuple[int]]] = []

        for token_id, context in tqdm.tqdm(
            token_id_to_contexts, desc="Organizing tokens into requests"
        ):
            for seq_id, (
                recorded_token_ids,
                full_context,
                orig_context,
            ) in enumerate(unique_sequences):
                if context == full_context:
                    # same request
                    new_recorded_token_ids = recorded_token_ids + [token_id]
                    unique_sequences[seq_id] = (
                        new_recorded_token_ids,
                        full_context + [token_id_to_output_token[token_id]],
                        orig_context,
                    )
                    break
            else:
                # new request
                unique_sequences.append(
                    (
                        [token_id],
                        context + [token_id_to_output_token[token_id]],
                        context,
                    )
                )

        print(f"Found {len(unique_sequences)} unique sequences.")
        with open(os.path.join(dataset_dir, "dataset.pkl"), "wb") as f:
            pickle.dump(
                (
                    unique_sequences,
                    token_id_to_output_token,
                    dict(token_id_to_experts),
                    n_layers,
                ),
                f,
            )
    else:
        with open(os.path.join(dataset_dir, "dataset.pkl"), "rb") as f:
            (
                unique_sequences,
                token_id_to_output_token,
                token_id_to_experts,
                n_layers,
            ) = pickle.load(f)

    unique_sequences = unique_sequences[:max_samples]
    if max_layers:
        n_layers = min(n_layers, max_layers)

    # repeat unique sequences
    unique_sequences = unique_sequences * repeat

    # cpp simulator requires:
    # 1. list [req_id, token_id, layer_id, experts]
    # 2. list [req_id, context_tokens]
    # 3. list [req_id, output_tokens]
    expert_selection_list = []
    context_list = []
    output_list = []

    for req_id, (token_ids, _, orig_context) in tqdm.tqdm(
        enumerate(unique_sequences),
        total=len(unique_sequences),
        desc="Preparing request graphs",
    ):
        decoded_token_ids = [
            token_id_to_output_token[token_id] for token_id in token_ids
        ] * repeat_tokens
        if max_decoded_tokens:
            decoded_token_ids = decoded_token_ids[:max_decoded_tokens]
        per_token_experts = []
        for token_index, token_id in enumerate(token_ids * repeat_tokens):
            per_layer_experts = []
            for layer_id in range(n_layers):
                expert_ids = token_id_to_experts[token_id][layer_id]
                if k_experts_per_token is None:
                    k_experts_per_token = len(expert_ids)
                else:
                    assert k_experts_per_token == len(expert_ids), (
                        "Found different number of experts per token: "
                        "{} vs {}".format(k_experts_per_token, len(expert_ids))
                    )
                per_layer_experts.append(list(expert_ids))
            per_token_experts.append(per_layer_experts)
            if max_decoded_tokens and token_index == max_decoded_tokens - 1:
                break
        expert_selection_list.append(per_token_experts)
        context_list.append(orig_context * repeat_tokens)
        output_list.append(decoded_token_ids)
    return (
        expert_selection_list,
        context_list,
        output_list,
        n_layers,
        n_experts,
        k_experts_per_token,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str, required=True)
    parser.add_argument("-c", "--cost-model-dir", type=str, required=True)
    parser.add_argument("-s", "--strategy", type=str, required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--truncate-tokens", type=int, default=None)
    parser.add_argument("--repeat-tokens", type=int, default=1)
    parser.add_argument("--truncate-layers", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-candidates-per-expert", type=int, default=128)
    parser.add_argument("--per-token-latency-slo", type=float, default=1000.0)
    # 512 bytes per token per layer if TP = 8, 1024 if TP=4
    parser.add_argument("--per-token-kv-bytes", type=int, default=512)
    # 40GB - 47 * 2 / 8 GB = 28.25GB for KV Cache if TP = 8
    # 40GB - 47 * 2 / 4 GB = 16.5GB for KV Cache if TP = 4
    parser.add_argument("--memory-limit-mb", type=int, default=28250)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # scale memory limit if slicing layers
    if args.truncate_layers:
        args.memory_limit_mb = args.memory_limit_mb * args.truncate_layers / 32
    # print args
    print("Args:")
    for arg in vars(args):
        print("\t{}: {}".format(arg, getattr(args, arg)))
    return args


def main(args):
    (
        expert_selection_list,
        context_list,
        output_list,
        n_layers,
        n_experts,
        k_experts_per_token,
    ) = load_dataset(
        args.dataset_dir,
        args.n_samples,
        args.truncate_tokens,
        args.truncate_layers,
        args.repeat_tokens,
        args.repeat,
    )
    print("Loaded {} requests.".format(len(expert_selection_list)))

    cm_save_path = os.path.join(args.cost_model_dir, "cost_model.pkl")
    if os.path.exists(cm_save_path):
        cost_model = ProfileBasedCostModel.load(cm_save_path)
    else:
        cost_model = ProfileBasedCostModel(args.cost_model_dir)
        cost_model.save(cm_save_path)

    t = time.time()
    (
        stats,
        throughput,
        avg_cost_per_step,
        peak_kv_tokens,
        avg_act_experts,
        avg_bs,
    ) = simulator.run_simulation(
        cost_model,
        expert_selection_list,
        context_list,
        output_list,
        n_layers,
        n_experts,
        k_experts_per_token,
        args.max_batch_size,
        int(
            math.ceil(
                args.memory_limit_mb * 1e6 / args.per_token_kv_bytes / n_layers
            )
        ),
        args.per_token_latency_slo,
        args.strategy,
    )
    elapsed = time.time() - t
    print("")
    print("Finished {} requests in {}s.".format(len(stats), elapsed))
    flattened_token_latencies = [
        latency
        for stat in stats.values()
        for latency in stat.get_per_token_latencies()[1:]
    ]
    print("Avg activated experts per batch: {}".format(avg_act_experts))
    print("Avg batch size: {}".format(avg_bs))
    print(
        "Avg latency: {} ms.".format(
            sum(flattened_token_latencies) / len(flattened_token_latencies)
        )
    )
    print("Avg throughput: {} tokens/s.".format(throughput))
    print("Avg cost per step: {} ms.".format(avg_cost_per_step))
    print("Peak KV tokens: {}.".format(peak_kv_tokens))
    print(
        "Peak KV tokens: {} / {} MB".format(
            peak_kv_tokens * args.per_token_kv_bytes * n_layers / 1e6,
            args.memory_limit_mb,
        )
    )


if __name__ == "__main__":
    main(parse_args())
