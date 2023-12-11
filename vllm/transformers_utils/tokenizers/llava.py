from transformers import PreTrainedTokenizerBase, PretrainedConfig
from transformers.utils import logging

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

logger = logging.get_logger(__name__)

def llava_add_special_tokens(config: PretrainedConfig,
                             tokenizer: PreTrainedTokenizerBase):
    """Add llava special tokens to the tokenizer."""
    mm_use_im_start_end = getattr(config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    return tokenizer



