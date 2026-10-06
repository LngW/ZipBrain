#!/bin/bash

suffix=${HOST:-${NAME}}
if ((${#suffix} == 0)); then
    echo No suffix available, exiting
    exit
fi

common_args=(
    run_optuna_grid_full_m3.py
    --num_workers 16
    --test
    --batch_size ${2:-'128'}
    --model TFM
    --workspace optuna_tfm
    --namespace tfm_tpe8_1f
    --storage sqlite:///./workspace/optuna_tfm/example.$(uname -n).sqlite3
    # --n_trials_qmc 0
    # --skip-s1
    # --skip-s2
    --skip-s3
    --skip-s4
    --seed 42
    # --suffix fml1
    # --fml1
)

common_args="${common_args[@]}"

# ToMe doi=10.48550/arXiv.2210.09461: Fixed hyperparameter r and gradually reduce tokens layer by layer
# ToFU doi=10.1109/WACV57701.2024.00141: Similar to ToMe, but retro to prune, and introduced mlerp as a replaceement of slerp
# clsp = [cls](prune) doi=10.48550/arXiv.2412.01818: Use attention score with [cls] to prune unimportant tokens
# meanp: EEG models may not use a [cls] token, so use mean of tokens as a replacement of [cls]. This approach depends on the cornve space
# clstrpts (TR-PTS, a merging version of [cls] pruning): 
# meantrpts (TR-PTS, a merging version of mean pruning): 
# methods=(tome tofu clsp meanp clstrpts meantrpts) 
methods0=(
    # 'kiddp[k]' 'kiddp[q]' 'kiddp[v]'
    # 'kiddl2[k]' 
    # 'kiddl2[q]'
    # 'kiddl2[v]'
    # 'kiddl2pte[k]' 
    # 'kiddl2pte[q]' 
    # 'kiddl2pte[v]' 
    # 'kiddl2f{}[k]'
    # 'kiddl2f{}[q]'
    # 'kiddl2f{}[v]'
    # 'kiddl2f{adjust}[k]'
    # 'kiddl2f{bottom_pivot}[k]'
    # 'kiddl2f{adjust,bottom_pivot}[k]'
    # 'kiddl2f{adjust,merge_scheme=avglen}[k]'

    # 'kiddl2f{softmax_qk,adjust}[k]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k]'
    # 'kiddl2f{softmax_qk,adjust}[q]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[q]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[q]'
    # 'kiddl2f{softmax_qk,adjust}[v]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[v]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v]'
    # 'kiddl2f{softmax_qk,adjust}[vc,x,qh,qh,kh]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[vc,x,qh,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[vc,x,qh,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,merge_scheme=avglen}[k]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,qh,qh,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kh,qh,qh,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,q,q,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kh,q,q,x,x]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,qh,qh,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,k,k,v,v]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,x,k,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,q,k,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,v,k,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,k,q,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,k,v,k,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[x,k,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[x,q,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[x,v,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[x,x,x,x,x]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,kh,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kc,kh,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kh,k,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kc,k,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[k,kc,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[kh,kc,x,x,x]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qh,qh,x,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[vh,vh,x,x,x]'
    
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,q,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,k,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kc,x,x]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,v,x,x]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,q,q]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,q,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,q,v]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qh,qh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qh,vh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qc,qc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qc,kc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qc,qc,kh,qc,vc]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,q,q]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,q,k]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,q,v]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qh,qh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qh,vh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qc,qc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qc,kc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qc,vc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,x,x]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qc,kc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,kh,qc,kc]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,vc,kh,qc,kc]'

    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[qh,qh,kh,qh,qh]'

    # 'kiddl2f{adjust}[vc,x,k,qh,kh]'
    # 'kiddl2f{bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{adjust,bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{adjust,merge_scheme=avglen}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[q]'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v]'
    # 'kiddl2f{softmax_qk,adjust,merge_scheme=avglen}[vc,x,k,qh,kh]'

    # 'kiddl2f{softmax_qk,bottom_pivot,norm_imp=false}[k]'
    # 'kiddl2f{softmax_qk,bottom_pivot,norm_imp=false}[k]'
    # 'kiddl2f{bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,bottom_pivot}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,bottom_pivot,merge_scheme=avglen}[vc,x,k,qh,kh]'
    # 'kiddl2f{softmax_qk,bottom_pivot,norm_imp=false}[vc,x,k,qh,kh]'
    'kiddl2f{}'
    'kiddl2f{bottom_pivot}'
    # 'kiddl2f{adjust}'
    # 'kiddl2f{softmax_qk}'
    # 'kiddl2f{bottom_pivot}'
    'kiddl2f{adjust,bottom_pivot}'
    # 'kiddl2f{softmax_qk,bottom_pivot}'
    'kiddl2f{softmax_qk,adjust,bottom_pivot}'
    'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
)

# for tpe6: which of bottom_pivot or avglen works better?
# those are designs used to avoid conflict of mlerp and norm-based pivot selection.
methods0=(
    'kiddl1s{norm_imp=false}'

    # 'kiddl2f{}'
    # 'kiddl2f{norm_imp=false}'
    # 'kiddl2f{bottom_pivot,norm_imp=false}'
    # 'kiddl2f{adjust,bottom_pivot,norm_imp=false}'
    # 'kiddl2f{merge_scheme=avglen,norm_imp=false}'
    # 'kiddl2f{adjust,merge_scheme=avglen,norm_imp=false}'
    # 'kiddl2f{adjust,norm_imp=false}'
    # 'kiddl2f{softmax_qk,norm_imp=false}'
    # 'kiddl2f{adjust,norm_imp=false}'
    # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
)


datasets=${1//;/ }
datasets=${datasets:-'TUAB TUEV ISRUC EEGMAT EarEEG'}

for ds in ${datasets}; do
    # methods=('kiddl2pte[k]')
    methods=(${methods0[@]})
    if [[ "$ds" == "TUAB" ]]; then
        # ds=TUAB
        # methods=(
        #     # 'kiddl2f{}'
        #     # 'kiddl2f{norm_imp=false}'
        #     # 'kiddl2f{bottom_pivot,norm_imp=false}'
        #     # 'kiddl2f{adjust,bottom_pivot,norm_imp=false}'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,vh,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,qc,kc]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,kh,kc,qc]'
        #     # 'kiddl2f{norm_imp=false}'
        #     'kiddl2f{}'
        #     'kiddl2f{bottom_pivot}'
        #     'kiddl2f{adjust}'
        #     'kiddl2f{softmax_qk}'
        #     'kiddl2f{bottom_pivot}'
        #     'kiddl2f{adjust,bottom_pivot}'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot}'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
        # )
        for m in "${methods[@]}"; do
            # for r in 57; do
            # for r in 1 19 38 57 61 75; do
            # for r in 75; do
            # for r in 15 30 46; do # 1/5
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            # for r in 61; do
            # for r in 61 75; do
            for r in 75; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "TUEV" ]]; then
        # ds=TUEV
        # methods=('kiddl2[k]')
        # methods=(
        #     # stage 1 find best m0 & m1
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,q,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,k,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,v,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qc,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kc,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vc,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qh,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kh,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vh,x,x,x]'

        #     # stage 1.1 find m0 basing on m1
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,v,x,x,x]'


        #     # stage 2.1: find best m2
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,x,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,qh,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,kh,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,vh,x,x]'

        #     # stage 2.2
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,kc,x,x]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,x,x]'

        #     # stage 3
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,q,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,q,q]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,q,v]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,k,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,k,q]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,k,v]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,v,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,v,q]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,v,v]'

        #     # stage 3
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,x,q,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,x,qh,kh]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,qh,q,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,qh,qh,kh]'

        # )
        # # methods=(
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,q,k]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,q,q]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,q,v]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,k,k]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,k,q]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,k,v]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,v,k]'
        # #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,v,q]'
        # #     'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,v,k,vh,qh]'
        # #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,vh,qh]'
        # #     'kiddl2f{softmax_qk,adjust,norm_imp=false}[v,v,k,vh,qh]'
        # #     'kiddl2f{softmax_qk,bottom_pivot,norm_imp=false}[v,v,k,vh,qh]'
        # #     'kiddl2f{adjust,bottom_pivot,norm_imp=false}[v,v,k,vh,qh]'
        # # )
        # # methods=(
        # #     # 'kiddl2f{}[k]'
        # #     # 'kiddl2f{}[x,x,x,x,x]'
        # #     'kiddl2f{bottom_pivot}[k,k,x,x,x]'
        # #     'kiddl2f{bottom_pivot}[q,q,x,x,x]'
        # #     'kiddl2f{bottom_pivot}[v,v,x,x,x]'
        # # )
        # methods=(
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
        # )
        # methods=(
        #     'kiddl2f{}'
        #     'kiddl2f{norm_imp=false}'
        #     'kiddl2f{bottom_pivot}'
        #     'kiddl2f{adjust}'
        #     'kiddl2f{softmax_qk}'
        #     'kiddl2f{bottom_pivot,norm_imp=false}'
        #     'kiddl2f{adjust,bottom_pivot,norm_imp=false}'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}'
        # )
        for m in "${methods[@]}"; do
            # for r in 9; do
            # for r in 1 9 18 27 29 35; do
            # for r in 35; do
            # for r in 7 14 22; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            # for r in 29 35; do
            for r in 35; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EEGMAT" ]]; then
        # ds=EEGMAT
        # methods=('kiddl2[k]')
        # methods=(
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,x,x,x,x]'

        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,q,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,qh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,qc,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,kh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,kc,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,vh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,vc,x,x,x]'

        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,x,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,x,x,x,x]'

        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,q,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qc,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,k,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kc,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,v,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vc,x,x,x]'
        # )
        # methods=(
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,vh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,vh,x,x,x]'

        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,qh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,kh,x,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,vh,x,x,x]'
        # )
        # methods=(
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,qh,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,vh,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,k,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,q,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,v,x,x]'
        # )
        # methods=(
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,q,k]'
        #     # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,qh,kh]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,q,q]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,qh,qh]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,k,k]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,kh,kh]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,v,v]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,vh,vh]'
        # )
        # methods=(
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,x,x]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,q,k]'
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,kh,qh,kh]'
        # )
        for m in "${methods[@]}"; do
            # for r in 1 6 11 17 22 27; do
            # for r in 27; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 27; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "ISRUC" ]]; then
        # ds=ISRUC
        # methods=('kiddl2[q]' 'kiddl2[v]')
        # methods=('kiddl2pte[q]' 'kiddl2pte[v]')
        # methods=(
            # 1.0 three types of m1 can be selected: same as m0, v[c|h], or x
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,q,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,v,x,x,x]'

            # 'kiddl2f{}[x,x,x,x,x]'
            # 'kiddl2f{}[q,x,x,x,x]'
            # 'kiddl2f{}[k,x,x,x,x]'
            # 'kiddl2f{}[v,x,x,x,x]'
            # 'kiddl2f{}[x,q,x,x,x]'
            # 'kiddl2f{}[x,k,x,x,x]'
            # 'kiddl2f{}[x,v,x,x,x]'

            # 1.1
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qc,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kc,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vc,x,x,x]'


            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,vh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,vh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,vh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,k,x,x,x]'

            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,qh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,vh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,qc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,vc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,q,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,k,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,v,x,x]'

            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,qh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,kh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,vh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,qc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,kc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,vc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,k,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,q,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,v,x,x]'

            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,v,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,v,qh,kh]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kh,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kh,qh,kh]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kc,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,kc,kc,qh,kh]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,kh,v,qc,kc]'
        # )
        for m in "${methods[@]}"; do
            # for r in 1 18 35 53 71 88; do
            # for r in 88; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            # for r in 71 88; do
            for r in 88; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    elif [[ "$ds" == "EarEEG" ]]; then
        # ds=EarEEG
        # methods=('kiddl2[k]')
        # methods=(
            # 1.0
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[q,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vh,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[qc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[kc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[vc,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,q,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vh,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,qc,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,kc,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[x,vc,x,x,x]'

            # 1.1
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[k,k,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,k,x,x,x]'

            # 2.0
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,x,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,q,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,k,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,v,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,qh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,vh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,qc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kc,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,vc,x,x]'

            # 3.0
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,qh,kh]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,q,q]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,k,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,v,v]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,x,kh,x,x]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,x,kh,q,q]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,x,kh,k,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot}[v,x,kh,v,v]'

            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,kh,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,k,q,k]'
            # 'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,v,k,q,k]'

            # final
        #     'kiddl2f{softmax_qk,adjust,bottom_pivot,norm_imp=false}[v,x,kh,q,k]'
        # )
        for m in "${methods[@]}"; do
            # for r in 35 47; do
            # for r in 1 12 24 35 47 58; do
            # for r in 58; do
            #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
            # done
            for r in 58; do
                python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
            done
        done
    fi
done