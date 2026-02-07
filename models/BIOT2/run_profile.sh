#!/bin/bash
common_args=(
    --epochs 2
    --dataset RANDOM
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 64
    --model BIOT
    --num_workers 2
)
common_args="${common_args[@]}"
. ../../seeds

export CUDA_VISIBLE_DEVICES=0

for seed in $seeds; do
    python run_binary_supervised_profile.py $common_args --seed $seed --log_dir baseline_std
    python run_binary_supervised+profile.py $common_args --seed $seed --log_dir baseline_lin --linear
    for r in 38 19; do
        python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir tome_${r}_std
        python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir tome_${r}_lin --linear
    done
done