#!/bin/bash
common_args=(
    --epochs 2
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

# seeds=$seeds

for seed in $seeds; do
    # python run_binary_supervised.py $common_args --seed $seed --log_dir baseline
    # for k in 76 38 19; do
    #     python run_binary_supervised.py $common_args --seed $seed --top_k $k --log_dir sa_top_$k
    # done
    for r in 38 19; do
        python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir sa_tome_$r
        python run_binary_supervised.py $common_args --seed $seed --tome_r $r --linear --log_dir li_tome_$r
    done
done