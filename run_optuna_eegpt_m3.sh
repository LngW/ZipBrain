#!/bin/bash

suffix=${HOST:-${NAME}}
if ((${#suffix} == 0)); then
    echo No suffix available, exiting
    exit
fi

common_args=(
    run_optuna_grid_full_m3.py
    --num_workers 8
    --test
    --batch_size ${2:-'128'}
    --model EEGPT
    --workspace optuna_eegpt
    --namespace eegpt_tpe8_1f
    --storage sqlite:///./workspace/optuna_eegpt/example.$(uname -n).sqlite3
    # --skip-s1
    # --skip-s2
    --skip-s3
    --skip-s4
    # --n_trials_qmc 200
    # --n_trials_tpe 0
)

common_args="${common_args[@]}"

# ToMe doi=10.48550/arXiv.2210.09461: Fixed hyperparameter r and gradually reduce tokens layer by layer
# ToFU doi=10.1109/WACV57701.2024.00141: Similar to ToMe, but retro to prune, and introduced mlerp as a replaceement of slerp
# clsp = [cls](prune) doi=10.48550/arXiv.2412.01818: Use attention score with [cls] to prune unimportant tokens
# meanp: EEG models may not use a [cls] token, so use mean of tokens as a replacement of [cls]. This approach depends on the cornve space
# clstrpts (TR-PTS, a merging version of [cls] pruning): 
# meantrpts (TR-PTS, a merging version of mean pruning): 
# methods=(tome tofu clsp meanp clstrpts meantrpts) 
# methods0=(
#     # 'kiddp[k]' 'kiddp[q]' 'kiddp[v]'
#     # 'kiddl2[k]' 
#     # 'kiddl2[q]'
#     # 'kiddl2[v]'
#     'kiddl2pte[k]' 
#     'kiddl2pte[q]' 
#     'kiddl2pte[v]' 
# )

methods0=(
    'kiddl1s{}'
)

datasets=${1//;/ }
datasets=${datasets:-'TUAB TUEV ISRUC EEGMAT EarEEG'}

for ds in ${datasets}; do
    methods=("${methods0[@]}")
    if [[ "$ds" == "TUAB" ]]; then
        # ds=TUAB
        for m in "${methods[@]}"; do
            # for r in 1 2 3 4; do
            for r in 4; do
                python $common_args --dataset $ds --tome_variant $m --tome_r 0 $r 0 $r 0 $r 0 $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "TUEV" ]]; then
        # ds=TUEV
        for m in "${methods[@]}"; do
            # for r in 1 2 3 4; do
            for r in 4; do
                python $common_args --dataset $ds --tome_variant $m --tome_r 0 $r 0 $r 0 $r 0 $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EEGMAT" ]]; then
        # ds=EEGMAT
        for m in "${methods[@]}"; do
            # for r in 1 2 3; do
            for r in 3; do
                python $common_args --dataset $ds --tome_variant $m --tome_r 0 $r 0 $r 0 $r 0 $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "ISRUC" ]]; then
        # ds=ISRUC
        for m in "${methods[@]}"; do
            for r in 1; do
                python $common_args --dataset $ds --tome_variant $m --tome_r 0 $r 0 $r 0 $r 0 $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EarEEG" ]]; then
        # ds=EarEEG
        for m in "${methods[@]}"; do
            for r in 1; do
                python $common_args --dataset $ds --tome_variant $m --tome_r 0 $r 0 $r 0 $r 0 $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    fi
done