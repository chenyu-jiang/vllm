FROM_EXPERT_ID=$1

export CUDA_VISIBLE_DEVICES=$FROM_EXPERT_ID
cd ~/vllm/benchmarks/weight_explorer
for TO_EXPERT_ID in {0..7}
do
    # skip if the expert is the same
    if [ $FROM_EXPERT_ID -eq $TO_EXPERT_ID ]
    then
        continue
    fi
    for r in 128 512
    do
        for layer_id in {0..31}
        do
            out_dir=/root/lora_exps/from${FROM_EXPERT_ID}_to${TO_EXPERT_ID}_l${layer_id}_r${r}
            mkdir -p $out_dir
            python3 transfer.py --data_dir /root/expert_data --output_dir $out_dir --lora_alpha $r --r $r --from_expert_id $FROM_EXPERT_ID --to_expert_id $TO_EXPERT_ID --layer_id $layer_id
        done
    done
done