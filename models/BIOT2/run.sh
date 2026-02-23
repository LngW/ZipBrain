#!/bin/bash
workspace='comprehensive_lin_896_5e-10_dropout5_l'
workspace="workspace/${workspace}"
common_args=(
    # --epochs 2
    --dataset TUAB
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 896
    --lr 5e-4
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

export CUDA_VISIBLE_DEVICES=1

filename="run_binary_supervised.py"
# filename="-m debugpy --listen :6678 --wait-for-client $filename"

for seed in "${seeds[@]}"; do
    # baseline
    # python $filename $common_args --seed $seed --log_dir baseline_lin --linear
    # python $filename $common_args --seed $seed --log_dir baseline_std
    # python $filename $common_args --seed $seed --log_dir baseline_std_flash --flash

    # vanilla tome
    # for variant in "tome"; do
    #     # for r in "19 19 19 19" "38 38 38 38" "152 0 0 0" "152 76 0 0" "152 76 38 0" "152 76 38 19"; do
    #     for r in "19 19 19 19" "38 38 38 38"; do
    #         t=($r)
    #         t=($(for i in ${t[@]}; do echo $((i / 19)); done))
    #         t="${t[*]}"
    #         t=${t// /}

    #         python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
    #         # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
    #         # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
    #     done
    # done

    # for variant in "tomex"; do
    #     for r in "19 19 19 19" "38 38 38 38" "152 0 0 0" "152 76 0 0" "152 76 38 0" "152 76 38 19"; do
    #     # for r in "19 19 19 19" "38 38 38 38"; do
    #         t=($r)
    #         t=($(for i in ${t[@]}; do echo $((i / 19)); done))
    #         t="${t[*]}"
    #         t=${t// /}

    #         python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
    #         # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
    #         # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
    #     done
    # done

    # full (it has no qk version, only q version)
    # learnable (only x version, no k version)
    # for variant in "f_tome" "fx_tome" "l_tome" "lq_tome"; do
    for variant in "l_tome"; do
        for r in "19 19 19 19" "38 38 38 38" "285 0 0 0" "266 0 0 0" "228 0 0 0" "152 0 0 0" "152 76 0 0" "152 76 38 0" "152 76 38 19"; do
            t=($r)
            t=($(for i in ${t[@]}; do echo $((i / 19)); done))
            t="${t[*]}"
            t=${t// /}

            python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
            # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
            # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
        done
    done

    # full ch (on k only)
    # learnable ch (on x only, )
    # for variant in "fch_tome" "fxch_tome" "l_channel" "lq_channel"; do
    for variant in "l_channel"; do
        for r in "2 2 2 2" "1 1 1 1" "15 0 0 0" "14 0 0 0" "12 0 0 0" "8 0 0 0" "8 4 0 0" "8 4 2 0" "8 4 2 1"; do
            t=${r// /}

            python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
            # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
            # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
        done
    done

    # learnabel ch tome
    # for r in "12 3 0 0" "15 0 0 0" "8 0 0 0" "8 4 0 0" "8 4 2 0" "8 4 2 1"; do
    #     t=${r// /}
    #     variant=l_channel

    #     python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
    #     # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
    #     # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
    # done

    # channel
    # for r in "8 0 0 0" "8 4 0 0" "8 4 2 0" "8 4 2 1"; do
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_lin --linear
    #     # python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std
    #     # python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std_flash --flash
    # done

    # timestep
    # for r in "9 0 0 0" "9 5 0 0" "9 5 2 0" "9 5 2 1"; do
    #     python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_lin --linear
    #     # python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_std
    #     # python $filename $common_args --seed $seed --tome_variant time --tome_r $r --log_dir ts_tome_${r// /}_std_flash --flash
    # done

    # wch
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_lin --linear
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_std
    # python $filename $common_args --seed $seed --tome_variant w_channel w_channel --tome_r 8 2 --log_dir wch_tome_82_std_flash --flash

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

done