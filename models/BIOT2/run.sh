#!/bin/bash
workspace='ch_tome_ffn90_adamw_sbz100_cls_token_norm'
workspace="workspace/${workspace}"
common_args=(
    # --epochs 2
    --dataset TUAB
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 256
    --model BIOT
    --num_workers 8
    --subset 100
    --workspace $workspace
    --cls_token
)

if [ -d $workspace ]; then
    echo "workspace \"${workspace}\" exist, script exit"
    exit
fi
mkdir -p $workspace

common_args="${common_args[@]}"
. ../../seeds

export CUDA_VISIBLE_DEVICES=0

filename="run_binary_supervised.py"
# filename="-m debugpy --listen :6678 --wait-for-client $filename"

for seed in "${seeds[@]}"; do
    python $filename $common_args --seed $seed --log_dir baseline_lin --linear
    python $filename $common_args --seed $seed --log_dir baseline_std
    python $filename $common_args --seed $seed --log_dir baseline_std_flash --flash
    for r in 38 19; do
        python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_lin --linear
        python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_std
        python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_std_flash --flash
    done
    for r in 3 2 1; do
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_lin --linear
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_std
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_std_flash --flash
    done
    for r in "8 4 2 1" "8 4 2"; do
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_lin --linear
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std
        python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std_flash --flash
    done
done