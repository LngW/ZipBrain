#!/bin/bash
workspace='test_checkpoint'
workspace="workspace/${workspace}"
common_args=(
    # --epochs 2
    --dataset TUAB
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 64
    --model BIOT
    --num_workers 2
    --subset 100
    --workspace $workspace
    --test
    --no_train
    --load_from_checkpoint "workspace/profile_scatter_add/logs/tome_38_std_flash/RANDOM-BIOT-0.001-64-200-200-100-12345856/checkpoints/epoch=1-step=2.ckpt"
)

if [ -d $workspace ]; then
    echo "workspace \"${workspace}\" exist, script exit"
    exit
fi
mkdir -p $workspace

common_args="${common_args[@]}"
. ../../seeds

export CUDA_VISIBLE_DEVICES=0

for seed in $seeds; do
    # python run_binary_supervised.py $common_args --seed $seed --log_dir baseline_lin --linear
    # python run_binary_supervised.py $common_args --seed $seed --log_dir baseline_std
    # python run_binary_supervised.py $common_args --seed $seed --log_dir baseline_std_flash --flash
    for r in 38; do
        # python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir tome_${r}_lin --linear
        # python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir tome_${r}_std
        python run_binary_supervised.py $common_args --seed $seed --tome_r $r --log_dir tome_${r}_std_flash --flash
    done
done