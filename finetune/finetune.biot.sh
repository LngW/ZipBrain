#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

common_args=(
    --in_channels 18
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt
    --model BIOT
    --num_workers 8
)

common_args="${common_args[@]}"

# CLI to kickoff a finetune on TUAB
python run_binary_supervised.py $common_args --dataset TUAB --sample_length 10 --batch_size 512 --epoch 25 --lr 5e-4

# CLI to kickoff a finetune on TUEV
python run_multiclass_supervised.py $common_args --dataset TUEV  --n_classes 6 -sample_length 5 --batch_size 512 --epoch 25 --lr 5e-4

# CLI to kickoff a finetune on WorkLoad (EEGMAT)
python run_binary_supervised.py $common_args --dataset EEGMAT --sample_length 4 --batch_size 754 --epoch 25 --lr 5e-4

# CLI to kickoff a finetune on EarEEG
python run_multiclass_supervised.py $common_args --dataset EarEEG --n_classes 6 --sample_length 30 --batch_size 128 --epoch 25 --lr 2e-4

# CLI to kickoff a finetune on ISRUC
python run_multiclass_supervised.py $common_args --dataset ISRUC --n_classes 5 --sample_length 30 --batch_size 512 --epoch 25 --lr 5e-4

