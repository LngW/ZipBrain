#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

# CLI to kickoff a finetune on TUAB
python run_class_finetuning.py \
    --output_dir ./checkpoints/finetune_tuab_base/ \
    --log_dir ./log/finetune_tuab_base \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 256 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 1 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUAB \
    --disable_qkv_bias \
    --seed 0

# CLI to kickoff a finetune on TUEV
python run_class_finetuning.py \
    --output_dir ./checkpoints/finetune_tuev_base/ \
    --log_dir ./log/finetune_tuev_base \
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
    --dataset TUEV \
    --disable_qkv_bias \
    --seed 0