#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

common_args=(
    --model labram_base_patch200_200
    --finetune ./checkpoints/labram-base.pth
    --disable_rel_pos_bias
    --abs_pos_emb
    --disable_qkv_bias
    --seed 0
    --update_freq 1
    --warmup_epochs 1
    --save_ckpt_freq 5
    --drop_path 0.1
)

common_args="${common_args[@]}"

# CLI to kickoff a finetune on TUAB
python run_class_finetuning.py $common_args \
    --dataset TUAB \
    --output_dir ./checkpoints/finetune_tuab_base_256/ \
    --log_dir ./log/finetune_tuab_base_256 \
    --epochs 50 \
    --batch_size 256 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \

# CLI to kickoff a finetune on TUEV
python run_class_finetuning.py \
    --dataset TUEV \
    --output_dir ./checkpoints/finetune_tuev_base_256/ \
    --log_dir ./log/finetune_tuev_base_256 \
    --epochs 50 \
    --batch_size 256 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \

# CLI to kickoff a finetune on EEGMAT/WORKLOAD
python run_class_finetuning.py \
    --dataset WORKLOAD \
    --output_dir ./checkpoints/finetune_eegmat_base_256/ \
    --log_dir ./log/finetune_eegmat_base_256 \
    --epochs 2 \
    --batch_size 256 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \

# CLI to kickoff a finetune on EarEEG
python run_class_finetuning.py \
    --dataset EarEEG \
    --output_dir ./checkpoints/finetune_eareeg_base_256/ \
    --log_dir ./log/finetune_eareeg_base \
    --epochs 2 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \

# CLI to kickoff a finetune on ISRUC
python run_class_finetuning.py \
    --output_dir ./checkpoints/finetune_isruc_base_256/ \
    --log_dir ./log/finetune_isruc_base \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 256 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 1 \
    --epochs 2 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset ISRUC \
    --disable_qkv_bias \
    --seed 0