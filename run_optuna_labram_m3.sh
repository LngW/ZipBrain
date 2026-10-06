#!/bin/bash

common_args=(
    run_optuna_grid_full_m3.py
    --num_workers 8
    --test
    --batch_size ${2:-'128'}
    --model LaBraM
    --workspace optuna_labram
    --namespace labram_tpe8_1f
    --storage sqlite:///./workspace/optuna_labram/example.$(uname -n).sqlite3
    # --skip-s1
    # --skip-s2
    --skip-s3
    --skip-s4
    --seed 42
)

common_args="${common_args[@]}"

methods0=(
    'kiddl1s{norm_imp=false}'
    # 'kiddl1f{mh_amax,norm_imp=false}'
    # 'kiddl1f{bottom_pivot,norm_imp=false}'
    # 'kiddl1f{merge_scheme=avglen,norm_imp=false}'

    # 'kiddl2f{norm_imp=false}'
    # 'kiddl2f{adjust,norm_imp=false}'
    # 'kiddl2f{bottom_pivot,norm_imp=false}'
    # 'kiddl2f{merge_scheme=avglen,norm_imp=false}'
    # 'kiddl2f{adjust,bottom_pivot,norm_imp=false}'
    # 'kiddl2f{adjust,merge_scheme=avglen,norm_imp=false}'
)

datasets=${1//;/ }
datasets=${datasets:-'TUAB TUEV ISRUC EEGMAT EarEEG'}

for ds in ${datasets}; do
    methods=("${methods0[@]}")
    if [[ "$ds" == "TUAB" ]]; then
        for m in "${methods[@]}"; do
            # for r in 1; do
            # for r in 4 8 12 15 19; do
            # for r in 4 8 12 15; do
            # for r in 19; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 19; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "TUEV" ]]; then
        # ds=TUEV
        for m in "${methods[@]}"; do
            # for r in 4; do
            # for r in 1 2 4 6 8 9; do
            # for r in 2 4 6 8; do
            # for r in 9; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 9; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EEGMAT" ]]; then
        # ds=EEGMAT
        for m in "${methods[@]}"; do
            # for r in 1 2 3 4 5; do
            # for r in 5; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 5; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "ISRUC" ]]; then
        # ds=ISRUC
        for m in "${methods[@]}"; do
            # for r in 3 12; do
            # for r in 1 3 6 9 12 14; do
            # for r in 14; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 14; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EarEEG" ]]; then
        # ds=EarEEG
        for m in "${methods[@]}"; do
            # for r in 1 2 4 6 8 9; do
            for r in 9; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
        done
    fi
done