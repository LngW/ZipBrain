#!/bin/bash
common_args=(
    --dataset TUAB
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 128
    --model BIOT
    --num_workers 2
)
common_args="${common_args[@]}"
. ../../seeds

for seed in "${seeds[@]}"; do
    python run_binary_supervised.py $common_args --log_dir baseline
done