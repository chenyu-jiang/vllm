import subprocess

# for parallel_method in ["TP_EP", "DP_TP_EP"]:
for parallel_method in ["DP_TP_EP"]:
    for tp_size in [8, 4, 2, 1]:
        if parallel_method == "TP_EP":
            dp_size = 1
        else:
            dp_size = int(8 / tp_size)
        world_size = int(tp_size * dp_size)
        subprocess.run(
            f"HF_HOME=/nfs/hf_cache torchrun --nnodes=1 --nproc-per-node={world_size} benchmark_moe_layer.py --tp-size {tp_size} --dp-size {dp_size} --parallel-method {parallel_method}",
            shell=True,
        )