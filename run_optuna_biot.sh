#!/bin/bash

suffix=${HOST:-${NAME}}
if ((${#suffix} == 0)); then
    echo No suffix available, exiting
    exit
fi

common_args=(
    run_optuna_grid_full.py
    --num_workers 8
    --test
    --batch_size 128
    --model BIOT
    --workspace optuna_biot_l2_top
    --storage sqlite:///workspace/optuna_biot/db_l2_top.${suffix}.sqlite3
    --n_trials_qmc 0
    --n_trials_tpe 0
)

common_args="${common_args[@]}"

# ToMe doi=10.48550/arXiv.2210.09461: Fixed hyperparameter r and gradually reduce tokens layer by layer
# ToFU doi=10.1109/WACV57701.2024.00141: Similar to ToMe, but retro to prune, and introduced mlerp as a replaceement of slerp
# clsp = [cls](prune) doi=10.48550/arXiv.2412.01818: Use attention score with [cls] to prune unimportant tokens
# meanp: EEG models may not use a [cls] token, so use mean of tokens as a replacement of [cls]. This approach depends on the cornve space
# clstrpts (TR-PTS, a merging version of [cls] pruning): 
# meantrpts (TR-PTS, a merging version of mean pruning): 
# methods=(tome tofu clsp meanp clstrpts meantrpts) 
methods=(
    # 'kiddp[k]' 'kiddp[q]' 'kiddp[v]'
    'kiddl2[k]' 
    # 'kiddl2[q]'
    # 'kiddl2[v]'
)

datasets=${1//;/ }

for ds in ${datasets}; do
    if [[ "$ds" == "TUAB" ]]; then
        # ds=TUAB
        for m in "${methods[@]}"; do
            for r in 75; do
            # for r in 1 19 38 57 61 75; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 75; do
            # for r in 61 75; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "TUEV" ]]; then
        # ds=TUEV
        for m in "${methods[@]}"; do
            # for r in 1 9 18 27 29 35; do
            for r in 35; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            # for r in 29 35; do
            for r in 35; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EEGMAT" ]]; then
        # ds=EEGMAT
        for m in "${methods[@]}"; do
            # for r in 1 6 11 17 22 27; do
            for r in 27; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 27; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "ISRUC" ]]; then
        # ds=ISRUC
        for m in "${methods[@]}"; do
            # for r in 1 18 35 53 71 88; do
            for r in 88; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            # for r in 71 88; do
            for r in 88; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EarEEG" ]]; then
        # ds=EarEEG
        for m in "${methods[@]}"; do
            # for r in 1 12 24 35 47 58; do
            for r in 58; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 58; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    fi
done