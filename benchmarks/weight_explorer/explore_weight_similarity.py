import numpy as np

from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)


for name, param in hf_model_weights_iterator("mistralai/Mixtral-8x7B-v0.1"):
    if "layers.0" in name and "experts" in name and "w2" in name:
        expert_id = int(name.split(".")[5])
        # calc rank of the weight
        rank = np.linalg.matrix_rank(param.float().numpy())
        print(f"Expert {expert_id} rank: {rank}")