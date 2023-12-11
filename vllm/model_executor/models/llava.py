# coding=utf-8
# Adapted from
# https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py
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

"""Inference-only LLaVA encoder and adapter compatible with HuggingFace weights."""
import os
import re
import requests
from typing import Optional, List, Union, Tuple, Any

from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn

from transformers import LlamaConfig, AutoConfig, PreTrainedTokenizerBase, \
                         CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from vllm.utils import MMInputType
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)

_IMAGE_TOKEN_INDEX = -200

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, cache_dir=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(
                                self.vision_tower_name, cache_dir=cache_dir)
        self.vision_tower = CLIPVisionModel.from_pretrained(
                                self.vision_tower_name,
                                cache_dir=cache_dir)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

class LLaVAVisionEncoder(nn.Module):
    def __init__(self, config: LlavaConfig) -> None:
        super().__init__()

        self.config = config

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    @property
    def dtype(self):
        return self.get_vision_tower().dtype

    def forward(self, images):
        images = images.to(dtype=self.dtype)
        image_features = self.get_vision_tower()(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,):
        # load vision tower weight from hf
        vision_tower = self.get_vision_tower()
        if vision_tower is not None:
            vision_tower: CLIPVisionTower
            vision_tower.load_model(cache_dir=cache_dir)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # load mm_projector weight
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision,
                name_filter=lambda name: name.startswith("mm_projector")):
            if "mm_projector" in name and name.startswith("model"):
                name = name.replace("model.", "")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

def _load_image(image_file: str):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def _expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def _process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = _expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def llava_tokenize_and_postprocess_fn(prompt: str,
                                      tokenizer: PreTrainedTokenizerBase):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [_IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])

    return input_ids


def get_llava_preprocess_and_collate_fn(model: LLaVAVisionEncoder):
    def _prep_and_collate(images: List[Union[str, Image.Image]]):
        """Preprocesses the images and collates them into a batch."""
        image_objs = []
        for image in images:
            if isinstance(image, str):
                image = _load_image(image)
            assert isinstance(image, Image.Image)
            image_objs.append(image)
        image_tensor = _process_images(image_objs,
                                    model.get_vision_tower().image_processor,
                                    model.config)
        image_tensor = image_tensor.to("cuda")
        return image_tensor

    return _prep_and_collate

def llava_prompt_mixer(prompt_ids: List[int],
                       prompt_embs: torch.Tensor,
                       multimodal_embs: List[Tuple[MMInputType, Any]]
                       ) -> torch.Tensor:
    num_images = 0
    image_token_indices = [-1]
    for n in prompt_ids:
        if n == _IMAGE_TOKEN_INDEX:
            num_images += 1
            image_token_indices.append(n)
    image_token_indices.append(len(prompt_ids))
    assert num_images == len(multimodal_embs), \
            ("Number of images in prompt does not match "
            "the number of multimodal inputs. "
            f"({num_images} vs {len(multimodal_embs)})")
    input_embs_noim = []
    for i in range(len(image_token_indices) - 1):
        input_embs_noim.append(prompt_embs[image_token_indices[i]+1:image_token_indices[i+1]])
    final_embs = []
    assert len(input_embs_noim) == len(multimodal_embs) + 1
    for i in range(len(input_embs_noim)):
        final_embs.append(input_embs_noim[i])
        if i < len(multimodal_embs):
            final_embs.append(multimodal_embs[i][1])
    return torch.cat(final_embs, dim=0)

AutoConfig.register("llava", LlavaConfig)