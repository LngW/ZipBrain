#!/bin/bash
workspace='test_consistance1'
# workspace="workspace/${workspace}"
common_args=(
    --epochs 2
    --dataset TUAB
    --in_channels 16
    --sampling_rate 200
    --token_size 200
    --hop_length 100
    --sample_length 10
    --batch_size 128
    --lr 1e-4
    # --weight_decay 1e-5
    --model BIOT
    --num_workers 4
    --subset 10
    --workspace $workspace
    # --cls_token
)

if [ -d "./workspace/$workspace" ]; then
    echo "workspace \"${workspace}\" exist, appending mode!"
    # rm -r ./workspace/$workspace
fi
mkdir -p "./workspace/$workspace"
cp $0 ./workspace/$workspace/$0

common_args="${common_args[@]}"
. ../../seeds

export CUDA_VISIBLE_DEVICES=0

filename="run_binary_supervised2.py"
# filename="-m debugpy --listen :6789 --wait-for-client $filename"

run_seed()
{
    variant=$1
    r=$2
    divider=$3

    t=($r)
    t=($(for i in ${t[@]}; do echo $((i / divider)); done))
    t="${t[*]}"
    t=${t// /}
    for seed in "${seeds[@]}"; do
        # echo $variant $t $r $seed
        python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
        # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
        # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
        break
    done
}

run_seed "l_tome" "285 0 0 0" 19
exit

for variant in "l_tome"; do
    for r in "285 0 0 0" "266 0 0 0" "228 0 0 0" "152 0 0 0" "152 76 0 0" "152 76 38 0" "152 76 38 19"; do
        run_seed $variant "$r" 19
    done
done
exit

for seed in $seeds; do
    python $filename $common_args --seed $seed --log_dir baseline_lin --linear
    # python $filename $common_args --seed $seed --log_dir baseline_lin --linear --pretrain_model_path workspace/test_tome_variants/logs/baseline_lin/TUAB-BIOT-0.001-64-200-200-100-12345856/checkpoints/epoch=4-step=160.ckpt
    # python $filename $common_args --seed $seed --log_dir baseline_lin --linear # --pretrain_model_path workspace/test_tome_variants/logs/baseline_lin/TUAB-BIOT-0.001-64-200-200-100-12345856/checkpoints/epoch=4-step=160.ckpt
    # python $filename $common_args --seed $seed --log_dir baseline_std
    # python $filename $common_args --seed $seed --log_dir baseline_std_flash --flash
    # for r in 38 19; do
    #     python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_lin --linear
    #     python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_std
    #     python $filename $common_args --seed $seed --tome_variant tome --tome_r $r --log_dir tome_${r}_std_flash --flash
    # done
    # for r in 3 2; do
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_lin --linear
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_std
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r}_std_flash --flash
    # done
    # for r in "8 4 2 1" "8 4 2"; do
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_lin --linear
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std
    #     python $filename $common_args --seed $seed --tome_variant channel --tome_r $r --log_dir ch_tome_${r// /}_std_flash --flash
    # done
done