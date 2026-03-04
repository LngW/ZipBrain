#!/bin/bash
workspace='learnable_merge2_896_5e-4'
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
        # echo $variant $r $t $seed
        python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_lin --linear
        # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std
        # python $filename $common_args --seed $seed --tome_variant ${variant} --tome_r $r --log_dir ${variant}_${t}_std_flash --flash
    done
}

# run_seed "baseline" 0 1
for r in "285 0 0 0" "266 0 0 0" "228 0 0 0" "152 0 0 0" "152 76 0 0" "152 76 38 0" "152 76 38 19"; do
    for variant in "l_tome_2_0" "l_tome_2_1" "l_tome_2s_0" "l_tome_2s_1"; do
        run_seed $variant "$r" 19
    done
done
exit
