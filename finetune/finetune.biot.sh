#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

# CLI to kickoff a finetune on TUAB
python run_binary_supervised.py --dataset TUAB --in_channels 18 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 10 --batch_size 512 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt --epoch 1 --lr 5e-4 --num_workers 8

# CLI to kickoff a finetune on WorkLoad (EEGMAT)
python run_binary_supervised.py --dataset EEGMAT --in_channels 18 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 4 --batch_size 1024 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt --epoch 10 --lr 5e-4 --num_workers 8

# CLI to kickoff a finetune on TUEV
python run_multiclass_supervised.py --dataset TUEV --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 5 --batch_size 512 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt --epoch 3 --lr 5e-4 --num_workers 8

# CLI to kickoff a finetune on EarEEG
python run_multiclass_supervised.py --dataset EarEEG --in_channels 18 --n_classes 6 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 30 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt --epoch 10 --lr 2e-4 --num_workers 8

# CLI to kickoff a finetune on ISRUC
python run_multiclass_supervised.py --dataset ISRUC --in_channels 18 --n_classes 5 --sampling_rate 200 --token_size 200 --hop_length 100 --sample_length 30 --batch_size 128 --model BIOT --pretrain_model_path pretrained-models/EEG-SHHS+PREST-18-channels.ckpt --epoch 10 --lr 5e-4 --num_workers 8

