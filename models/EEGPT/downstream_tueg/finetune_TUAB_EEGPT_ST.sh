#!/usr/bin/env bash
common_args=(
    --output_dir ./checkpoints/finetune_tuab_eegpt/
    --model EEGPT
    --weight_decay 0.05
    --batch_size 100
    --lr 5e-4
    --update_freq 1
    --warmup_epochs 5
    --epochs 25
    --layer_decay 0.65
    --dist_eval
    --save_ckpt_freq 5
    --disable_rel_pos_bias
    --abs_pos_emb
    --dataset TUAB
    --disable_qkv_bias
)
common_args="${common_args[@]}"

# Use same set of seeds for evaluation
. ../../../seeds

log_dir="./log/no_finetune"

for seed in "${seeds[@]}"; do
    folder_name="tuab_tiny_${seed}_baseline"
    python run_class_finetuning_EEGPT_change.py $common_args \
        --output_dir ./checkpoints/$folder_name \
        --log_dir $log_dir/$folder_name \
        --seed $seed

    for r in 2 1; do
        folder_name="tuab_tiny_${seed}_tome_${r}"
        python run_class_finetuning_EEGPT_change.py $common_args  \
            --output_dir ./checkpoints/$folder_name  \
            --log_dir $log_dir/$folder_name  \
            --seed $seed  \
            --tome_r $r
    done

    for k in 4 2 1; do
        folder_name="tuab_tiny_${seed}_top_${k}"
        python run_class_finetuning_EEGPT_change.py $common_args  \
            --output_dir ./checkpoints/$folder_name  \
            --log_dir $log_dir/$folder_name  \
            --seed $seed  \
            --top_k $k
    done
done
