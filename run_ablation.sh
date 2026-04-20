#!/bin/bash

common_args=(
    --model BIOT
    --dataset TUEV
    --storage sqlite:///workspace/ablation_biot/db.splite3
    --workspace workspace/ablation_biot
    --test
)

common_args="${common_args[@]}"

# echo ${#1}
# echo 
# echo ${#2}
# echo ${2//,/ }
# echo ${3//,/ }
# echo ${#3}

if ((${#1} > 0)); then
    variants=${1//,/ }
else
    echo variants are not specified!
    exit
fi

if ((${#2} > 0)); then
    metrics=${2//,/ }
else
    echo metrics are not specified!
    exit
fi

if ((${#3} > 0)); then
    rs=${3//,/ }
else
    echo rs are not specified!
    exit
fi

for r in $rs; do
    for me in $metrics; do
        for variant in $variants; do
            echo run_ablation.py $common_args --tome_r $r --tome_variant ${variant}[${me}] --log_dir biot-tuev-${variant}[${me}]-$r-full
            # python run_ablation.py $common_args --tome_r $r --log_dir biot-tuev-${variant}[${me}]-$r-full
        done
    done
done
