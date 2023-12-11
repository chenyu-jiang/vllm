"""Utilities for selecting and loading models."""
import contextlib
from typing import Type, Tuple, Callable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.utils import is_hip
from vllm.logger import init_logger

logger = init_logger(__name__)

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "AquilaForCausalLM": AquilaForCausalLM,  # AquilaChat2
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "ChatGLMModel": ChatGLMForCausalLM,
    "ChatGLMForConditionalGeneration": ChatGLMForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "LlavaLlamaForCausalLM": LlamaForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    # transformers's mpt class has lower case
    "MptForCausalLM": MPTForCausalLM,
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
    "YiForCausalLM": YiForCausalLM,
}

_NAME_FILTER_REGISTRY = {
    "LlavaLlamaForCausalLM": lambda x: not x.startswith("mm_projector"),
}

_ENCODER_REGISTRY = {
    "LlavaLlamaForCausalLM": LLaVAVisionEncoder,
}

_TOKENIZE_POSTPROCESS_FN_REGISTRY = {
    "LlavaLlamaForCausalLM": llava_tokenize_and_postprocess_fn,
}
_PREPROCESS_COLLATE_FN_REGISTRY = {
    "LlavaLlamaForCausalLM": get_llava_preprocess_and_collate_fn,
}
_PROMPT_MIXER_FN_REGISTRY = {
    "LlavaLlamaForCausalLM": llava_prompt_mixer,
}

# Models to be disabled in ROCm
_ROCM_UNSUPPORTED_MODELS = []
if is_hip():
    for rocm_model in _ROCM_UNSUPPORTED_MODELS:
        del _MODEL_REGISTRY[rocm_model]

# Models partially supported in ROCm
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    "MistralForCausalLM":
    "Sliding window attention is not supported in ROCm's flash attention",
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            if is_hip() and arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"{arch} is not fully supported in ROCm. Reason: "
                    f"{_ROCM_PARTIALLY_SUPPORTED_MODELS[arch]}")
            return _MODEL_REGISTRY[arch]
        elif arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model architecture {arch} is not supported by ROCm for now. \n"
                f"Supported architectures {list(_MODEL_REGISTRY.keys())}")
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")

def _get_name_filter_fn(config: PretrainedConfig) -> Callable[[str], bool]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NAME_FILTER_REGISTRY:
            return _NAME_FILTER_REGISTRY[arch]
    return lambda x: True

def _get_encoder_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _ENCODER_REGISTRY:
            return _ENCODER_REGISTRY[arch]
    raise ValueError(
        f"Encoder model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_ENCODER_REGISTRY.keys())}")

def _get_tokenize_and_postprocess_fn(config: PretrainedConfig):
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _TOKENIZE_POSTPROCESS_FN_REGISTRY:
            return _TOKENIZE_POSTPROCESS_FN_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_TOKENIZE_POSTPROCESS_FN_REGISTRY.keys())}")

def _get_preprocess_and_collate_fn(config: PretrainedConfig):
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _PREPROCESS_COLLATE_FN_REGISTRY:
            return _PREPROCESS_COLLATE_FN_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_PREPROCESS_COLLATE_FN_REGISTRY.keys())}")

def _get_prompt_mixer_fn(config: PretrainedConfig):
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _PROMPT_MIXER_FN_REGISTRY:
            return _PROMPT_MIXER_FN_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_PROMPT_MIXER_FN_REGISTRY.keys())}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            name_filter = _get_name_filter_fn(model_config.hf_config)
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision,
                               name_filter)
    return model.eval()

def get_multimodal_encoder(model_config: ModelConfig
                           ) -> Tuple[nn.Module, Callable, Callable, Callable]:
    model_class = _get_encoder_architecture(model_config.hf_config)

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_config)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
        tokenize_fn = _get_tokenize_and_postprocess_fn(model_config.hf_config)
        collate_fn = _get_preprocess_and_collate_fn(model_config.hf_config)(model)
        mixer_fn = _get_prompt_mixer_fn(model_config.hf_config)
    return model.eval(), tokenize_fn, collate_fn, mixer_fn
