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
    --model LaBraM
    --workspace optuna_labram_l2_top
    --storage sqlite:///workspace/optuna_labram/db_l2_top.${suffix}.sqlite3
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
            # for r in 1 4 8 12 15 19; do
            for r in 19; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 19; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "TUEV" ]]; then
        # ds=TUEV
        for m in "${methods[@]}"; do
            # for r in 1 2 4 6 8 9; do
            for r in 9; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 9; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EEGMAT" ]]; then
        # ds=EEGMAT
        for m in "${methods[@]}"; do
            # for r in 1 2 3 4 5; do
            for r in 5; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
            for r in 5; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "ISRUC" ]]; then
        # ds=ISRUC
        for m in "${methods[@]}"; do
            # for r in 1 3 6 9 12 14; do
            for r in 14; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            done
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