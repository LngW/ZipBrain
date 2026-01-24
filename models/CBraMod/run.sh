#!/bin/bash

common_args=(
    --cuda 0
    --epochs 25
    --batch_size 64
    --lr 5e-4
    --weight_decay 0.05
    --downstream_dataset TUAB
    --datasets_dir ./datas/tuab/v3.0.1/edf/100/process_refine
    --num_of_classes 1
    --num_workers 4
    # --use_pretrained_weights ""
)
common_args="${common_args[@]}"

echo $common_args

cp_dir="checkpoints"
log_dir="log"

. ../../seeds

for seed in "${seeds[@]}"; do
    python ./finetune_main.py $common_args \
        --model_dir ./$cp_dir/tuab_base_baseline \
        --seed $seed \
        > ./$log_dir/tuab_base_"${seed}"_baseline.log
    for k in 2 1; do
        folder_name="tuab_base_${seed}_top_${k}_t"
        python ./finetune_main.py $common_args  \
            --model_dir ./$cp_dir/$folder_name  \
            --seed $seed  \
            --top_k_t $k > \
            ./$log_dir/$folder_name.log
    done
    for k in 4 2 1; do
        folder_name="tuab_base_${seed}_top_${k}_s"
        python ./finetune_main.py $common_args  \
            --model_dir ./$cp_dir/$folder_name  \
            --seed $seed  \
            --top_k_s $k > \
            ./$log_dir/$folder_name.log
    done
done
