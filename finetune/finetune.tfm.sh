#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

common_args=(
    --vqvae_pretrained_path 'pretrained_weigths/multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth'
    --mem_pretrained_path 'pretrained_weigths/multiple_dataset_settings/MTP_Pretrained_tfm_encoder_64x4/tfm_encoder_mtp_last.pth'
    --gpu 0
    --num_workers 4
)

common_args="${common_args[@]}"

# CLI to kickoff a finetune on EarEEG
python downstream_transformer_finetuning.py $common_args \
    --dataset_name EarEEG \
    --save_path ./log/finetune_eareeg_256 \
    --batch_size 256

# CLI to kickoff a finetune on EEGMAT / WORKLOAD
python downstream_transformer_finetuning.py $common_args \
    --dataset_name WORKLOAD \
    --save_path ./log/finetune_eegmat_377 \
    --batch_size 377

# CLI to kickoff a finetune on ISRUC
python downstream_transformer_finetuning.py $common_args \
   --dataset_name ISRUC \
   --save_path ./log/finetune_isruc_256 \
   --batch_size 256