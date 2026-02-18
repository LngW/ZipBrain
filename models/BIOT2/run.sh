#!/bin/bash
workspace='fixed_metrics_tome_vanilla_3'
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
    # --lr 1e-4
    --model BIOT
    --num_workers 8
    --subset 100
    --workspace $workspace
    # --cls_token
)

if [ -d $workspace ]; then
    echo "workspace \"${workspace}\" exist, script exit"
    exit
fi
mkdir -p $workspace
cp $0 $workspace/$0

common_args="${common_args[@]}"
. ../../seeds

export CUDA_VISIBLE_DEVICES=0

filename="run_binary_supervised.py"
# filename="-m debugpy --listen :6678 --wait-for-client $filename"

for seed in "${seeds[@]}"; do
    # baseline
    # python $filename $common_args --seed $seed --log_dir baseline_lin --linear
    # python $filename $common_args --seed $seed --log_dir baseline_std
    # python $filename $common_args --seed $seed --log_dir baseline_std_flash --flash

    # vanilla baseline
    # 16 -> 12
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 19 --log_dir tome_1111_lin --linear
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 19 --log_dir tome_1111_std
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 19 --log_dir tome_1111_std_flash --flash
    # 16 -> 8
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 38 --log_dir tome_2222_lin --linear
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 38 --log_dir tome_2222_std
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 38 --log_dir tome_2222_std_flash --flash
    # 16 -> 8
    python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 0 --log_dir tome_8000_lin --linear
    python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 0 --log_dir tome_8000_std
    python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 0 --log_dir tome_8000_std_flash --flash
    # 16 -> 4
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 --log_dir tome_8400_lin --linear
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 --log_dir tome_8400_std
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 --log_dir tome_8400_std_flash --flash
    # 16 -> 2
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 --log_dir tome_8420_lin --linear
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 --log_dir tome_8420_std
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 --log_dir tome_8420_std_flash --flash
    # 16 -> 1
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 19 --log_dir tome_8421_lin --linear
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 19 --log_dir tome_8421_std
    # python $filename $common_args --seed $seed --tome_variant tome --tome_r 152 76 38 19 --log_dir tome_8421_std_flash --flash

    # channel
    # for r in "8 4 2 1" "8 4 2 0" "8 4 0 0" "8 0 0 0"; do
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_lin --linear
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std_flash --flash
    # done

    # wch
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_std
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_std_flash --flash

    # timestep
    # for r in "9 5 2 1" "9 5 2 0" "9 5 0 0" "9 0 0 0"; do
    #     python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_lin --linear
    #     python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_std
    #     python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_std_flash --flash
    # done

    # wts
    # python $filename $common_args --seed $seed --tome_variant w_time w_time --tome_r 10 2 --log_dir wts_tome_1002_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_time w_time --tome_r 10 2 --log_dir wts_tome_1002_std
    # python $filename $common_args --seed $seed --tome_variant w_time w_time --tome_r 10 2 --log_dir wts_tome_1002_std_flash --flash

    # wch_wts
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel w_time w_time --tome_r 8 2 10 2 --log_dir wch_wts_821002_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel w_time w_time --tome_r 8 2 10 2 --log_dir wch_wts_821002_std
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel w_time w_time --tome_r 8 2 10 2 --log_dir wch_wts_821002_std_flash --flash

    # wts_wch
    # python $filename $common_args --seed $seed --tome_variant w_time w_time w_channel w_channel --tome_r 10 2 8 2 --log_dir wts_wch_100282_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_time w_time w_channel w_channel --tome_r 10 2 8 2 --log_dir wts_wch_100282_std
    # python $filename $common_args --seed $seed --tome_variant w_time w_time w_channel w_channel --tome_r 10 2 8 2 --log_dir wts_wch_100282_std_flash --flash

    # wch wch ts ts
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel time time --tome_r 8 2 9 5 --log_dir wts_wch_ts_ts_8295_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel time time --tome_r 8 2 9 5 --log_dir wch_wch_ts_ts_8295_std
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel time time --tome_r 8 2 9 5 --log_dir wts_wch_ts_ts_8295_std_flash --flash
done