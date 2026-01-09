#!/bin/bash
tag="test_10"
args="--epochs 2 --subset 10 --batch_size 128 --lr 1e-4 --num_workers 4"

models=('BIOT' 'CBIOT' 'SABIOT' 'T9BIOT' 'T19BIOT' 'ToMeBIOTr38' 'ToMeBIOTr19' 'ToMeCBIOTr38' 'ToMeCBIOTr19')
# models=( 'BIOT' )
seeds=(12345856 85875035 46812486 68486431 86435434 34525135)
# seeds=( 12345 )

for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        python experiments.py $args --model $model --seed $seed --tag $tag
    done
done