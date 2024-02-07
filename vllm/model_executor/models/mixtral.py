# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
import os
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from transformers import MixtralConfig

# from vllm._C import moe_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_src_rank,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_data_parallel_group,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]

LORA_DIR = os.environ.get("MIXTRAL_LORA_DIR", None)
LORA_FROM_EXPERT = os.environ.get("MIXTRAL_LORA_FROM_EXPERT", None)
LORA_TO_EXPERTS_PER_LAYER = os.environ.get("MIXTRAL_LORA_TO_EXPERTS_PER_LAYER", None)
LORA_MAX_LAYERS = os.environ.get("MIXTRAL_LORA_MAX_LAYERS", 32)
if LORA_FROM_EXPERT is not None:
    LORA_FROM_EXPERT = int(LORA_FROM_EXPERT)
if LORA_TO_EXPERTS_PER_LAYER is not None:
    LORA_TO_EXPERTS_PER_LAYER = [int(x) for x in LORA_TO_EXPERTS_PER_LAYER.split(",")]

class MixtralParallelism:
    DATA_EXPERT_PARALLEL = "data_expert_parallel"
    TENSOR_EXPERT_PARALLEL = "tensor_expert_parallel"


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha, dtype):
        super(LoRALinear, self).__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=dtype)
        self.scale = lora_alpha
        self.dtype = dtype

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale

    def load_lora_weight(self, state_dict):
        self.lora_A.weight.data.copy_(state_dict['lora_A.weight'])
        self.lora_B.weight.data.copy_(state_dict['lora_B.weight'])
        self.lora_A.cuda()
        self.lora_B.cuda()


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()
        self.lora_w1 = None
        self.lora_w2 = None
        self.lora_w3 = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        if self.lora_w1 is not None:
            w1_out += self.lora_w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        if self.lora_w3 is not None:
            w3_out += self.lora_w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        output_hidden_states, _ = self.w2(current_hidden_states)
        if self.lora_w2 is not None:
            output_hidden_states += self.lora_w2(current_hidden_states)
        return output_hidden_states

    def init_lora(self, state_dict, lora_alpha=None):
        for w_name in ["w1", "w2", "w3"]:
            w_A = state_dict[f"{w_name}.lora_A.weight"]
            w_B = state_dict[f"{w_name}.lora_B.weight"]
            r = w_A.shape[0]
            in_features = w_A.shape[1]
            out_features = w_B.shape[0]
            lora_alpha = lora_alpha if lora_alpha is not None else r
            setattr(self, f"lora_{w_name}", LoRALinear(in_features, out_features, r, lora_alpha, w_A.dtype))
            getattr(self, f"lora_{w_name}").load_lora_weight({
                "lora_A.weight": w_A,
                "lora_B.weight": w_B
            })


class MixtralTensorParallelMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1_w3 = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.w2 = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           reduce_results=False,
                                           linear_method=linear_method)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w1_w3(hidden_states)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x



class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        layer_idx: int = 0,
        parallel_method: str = MixtralParallelism.TENSOR_EXPERT_PARALLEL
    ):
        super().__init__()
        self.config = config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_rank = get_data_parallel_rank()
        self.dp_size = get_data_parallel_world_size()
        self.layer_idx = layer_idx
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.parallel_method = parallel_method
        assert self.parallel_method in [
            MixtralParallelism.DATA_EXPERT_PARALLEL,
            MixtralParallelism.TENSOR_EXPERT_PARALLEL
        ], f"Invalid parallel method: {self.parallel_method}"
        if self.parallel_method == MixtralParallelism.TENSOR_EXPERT_PARALLEL:
            if self.tp_size > self.num_total_experts:
                raise ValueError(
                    f"Tensor parallel size {self.tp_size} is greater than "
                    f"the number of experts {self.num_total_experts}.")

            # Split experts equally between tensor parallel ranks
            self.expert_indicies = np.array_split(range(
                self.num_total_experts), self.tp_size)[self.tp_rank].tolist()
            if not self.expert_indicies:
                raise ValueError(
                    f"TP Rank {self.tp_rank} has no experts assigned to it.")
            mixtral_mlp = MixtralMLP
        else:
            # Split experts equally between data parallel ranks
            # and then each expert is sharded across tensor parallel ranks.
            self.all_rank_expert_indicies = np.array_split(
                range(self.num_total_experts), self.dp_size)
            self.expert_indicies = self.all_rank_expert_indicies[self.dp_rank].tolist()
            if not self.expert_indicies:
                raise ValueError(
                    f"DP Rank {self.dp_rank} has no experts assigned to it.")
            mixtral_mlp = MixtralTensorParallelMLP

        self.experts = nn.ModuleList([
            mixtral_mlp(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       linear_method=linear_method)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        if LORA_DIR is not None and self.layer_idx < LORA_MAX_LAYERS:
            # route FROM_EXPERT to TO_EXPERT instead
            selected_experts[selected_experts == LORA_FROM_EXPERT] = LORA_TO_EXPERTS_PER_LAYER[self.layer_idx]

        if self.parallel_method == MixtralParallelism.TENSOR_EXPERT_PARALLEL:
            final_hidden_states = None
            for expert_idx in self.expert_indicies:
                expert_layer = self.experts[expert_idx]
                expert_mask = (selected_experts == expert_idx)
                expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                    keepdim=True)

                current_hidden_states = expert_layer(hidden_states).mul_(
                    expert_weights)
                if final_hidden_states is None:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)

            return tensor_model_parallel_all_reduce(final_hidden_states).view(
                batch_size, sequence_length, hidden_dim)
        else:
            # dispatch
            n_tokens = hidden_states.shape[0]
            token_locations = torch.zeros((n_tokens, self.top_k),
                                           dtype=torch.long,
                                           device=hidden_states.device)
            moe_ops.moe_gen_location(token_locations, selected_experts,
                                     self.top_k, n_tokens,
                                     self.num_total_experts)
            dispatched_hidden_states = torch.zeros(
                (self.num_total_experts, n_tokens * self.top_k, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device)
            moe_ops.moe_dispatch(dispatched_hidden_states, selected_experts,
                                    token_locations, hidden_states, self.top_k,
                                    n_tokens, hidden_dim, self.num_total_experts)
            # [DxLE = E, C, M]
            local_hidden_states = torch.empty((self.num_total_experts, n_tokens * self.top_k, hidden_dim),
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
            if self.tp_rank == 0:
                # 1. exchange hidden_states
                torch.distributed.all_to_all_single(local_hidden_states,
                                                    dispatched_hidden_states,
                                                    group=get_data_parallel_group())
                # 2. broadcast hidden_states across tensor parallel ranks
                torch.distributed.broadcast(local_hidden_states,
                                            src=get_tensor_model_parallel_src_rank(),
                                            group=get_tensor_model_parallel_group())
            else:
                # receive hidden_states
                torch.distributed.broadcast(local_hidden_states,
                                            src=get_tensor_model_parallel_src_rank(),
                                            group=get_tensor_model_parallel_group())
            # [DxLE, C, M] -> [D, LE, C, M]
            per_expert_hidden_states = [
                t.squeeze() for t in 
                torch.split(
                    local_hidden_states.reshape(
                        -1,
                        len(self.expert_indicies),
                        n_tokens * self.top_k,
                        hidden_dim
                    ),
                    1,
                    dim=1
                )
            ]
            expert_outputs = []
            for local_expert_idx, expert_idx in enumerate(self.expert_indicies):
                expert_layer = self.experts[expert_idx]
                expert_out = expert_layer(per_expert_hidden_states[local_expert_idx])
                expert_outputs.append(expert_out)
            # List[D, C, M] -> [D, LE, C, M] -> [DxLE, C, M]
            local_hidden_states = torch.cat([t.unsqueeze(1) for t in expert_outputs], dim=1)
            local_hidden_states = local_hidden_states.reshape(-1, n_tokens * self.top_k, hidden_dim)
            # reduce to tp rank 0
            torch.distributed.reduce(local_hidden_states,
                                     dst=get_tensor_model_parallel_src_rank(),
                                     group=get_tensor_model_parallel_group())
            if self.tp_rank == 0:
                # exchange local_hidden_states across data parallel ranks
                torch.distributed.all_to_all_single(dispatched_hidden_states,
                                                    local_hidden_states,
                                                    group=get_data_parallel_group())
                # broadcast hidden_states across tensor parallel ranks
                torch.distributed.broadcast(dispatched_hidden_states,
                                            src=get_tensor_model_parallel_src_rank(),
                                            group=get_tensor_model_parallel_group())
            else:
                # receive hidden_states
                torch.distributed.broadcast(dispatched_hidden_states,
                                            src=get_tensor_model_parallel_src_rank(),
                                            group=get_tensor_model_parallel_group())
            # gather
            gathered_hidden_states = torch.zeros((n_tokens, hidden_dim),
                                                  dtype=hidden_states.dtype,
                                                  device=hidden_states.device)
            moe_ops.moe_gather(gathered_hidden_states,
                                dispatched_hidden_states,
                                routing_weights.to(hidden_states.dtype),
                                selected_experts,
                                token_locations,
                                self.top_k,
                                n_tokens,
                                hidden_dim)
            return gathered_hidden_states.view(batch_size, sequence_length, hidden_dim)


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        layer_idx: int = 0,
        parallel_method: str = MixtralParallelism.TENSOR_EXPERT_PARALLEL,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            linear_method=linear_method)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           linear_method=linear_method,
                                            layer_idx=layer_idx,
                                           parallel_method=parallel_method)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        parallel_method: str = MixtralParallelism.TENSOR_EXPERT_PARALLEL,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, linear_method=linear_method,
                                layer_idx=layer_idx,
                                parallel_method=parallel_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], input_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_lora_weight(self):
        if LORA_DIR is None:
            return
        assert len(LORA_TO_EXPERTS_PER_LAYER) == len(self.layers)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx >= LORA_MAX_LAYERS:
                continue
            expert_indices = layer.block_sparse_moe.expert_indicies
            for expert_idx in expert_indices:
                expert_layer = layer.block_sparse_moe.experts[expert_idx]
                if LORA_TO_EXPERTS_PER_LAYER[layer_idx] == expert_idx:
                    lora_to_expert = LORA_TO_EXPERTS_PER_LAYER[layer_idx]
                    print("Loading LoRA weight for expert", expert_idx, "from expert", LORA_FROM_EXPERT)
                    state_dict = torch.load(os.path.join(LORA_DIR, f"from{LORA_FROM_EXPERT}_to{lora_to_expert}_l{layer_idx}_r128", "lora_model.pt"))
                    expert_layer.init_lora(state_dict, lora_alpha=128)

class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        parallel_method: str = MixtralParallelism.TENSOR_EXPERT_PARALLEL,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.parallel_method = parallel_method
        self.model = MixtralModel(config, linear_method,
                                  parallel_method=parallel_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        if self.parallel_method == MixtralParallelism.DATA_EXPERT_PARALLEL:
            stacked_params_mapping.extend([
                ("w1_w3", "w1", 0),
                ("w1_w3", "w3", 1),
            ])

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        self.model.load_lora_weight()
