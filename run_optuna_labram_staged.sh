#!/bin/bash

suffix=${HOST:-${NAME}}
if ((${#suffix} == 0)); then
    echo No suffix available, exiting
    exit
fi

common_args=(
    run_optuna_staged.py
    --num_workers 8
    --test
    --batch_size 128
    --model LaBraM
    --workspace optuna_labram_xqkv
    # --storage sqlite:///workspace/optuna_biot/db_xqkv.${suffix}.sqlite3
    --storage rqlite+pyrqlite://archfs00:14571
    # --n_trials_qmc 130
    --n_trials_tpe 0
    --suffix bottom
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
    # 'kiddl2[k]' 
    # 'kiddl2mh_pte' 
    'kiddl2mh' 
    'kiddl2pte' 
    'kiddl2' 
    # 'kiddl2mh_pte[q]'
    # 'kiddl2mh_pte[k]' 
    # 'kiddl2mh_pte[v]'
    # 'kiddl2mh_pte[v,x,k,q]'
)

datasets=${1//;/ }

for ds in ${datasets}; do
    if [[ "$ds" == "TUAB" ]]; then
        # ds=TUAB
        for m in "${methods[@]}"; do
            # for r in 75; do
            # for r in 1 19 38 57 61 75; do
                # python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 19; do
            # for r in 61 75; do
                KIDD_PIVOT_BOTTOM=1 python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
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