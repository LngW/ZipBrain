#!/bin/bash

common_args=(
    run_inference.py
    --num_workers 8
    --test
    --batch_size ${1:-'128'}
    --model LaBraM
    --workspace inference_labram
)

common_args="${common_args[@]}"

# ToMe doi=10.48550/arXiv.2210.09461: Fixed hyperparameter r and gradually reduce tokens layer by layer
# ToFU doi=10.1109/WACV57701.2024.00141: Similar to ToMe, but retro to prune, and introduced mlerp as a replaceement of slerp
# clsp = [cls](prune) doi=10.48550/arXiv.2412.01818: Use attention score with [cls] to prune unimportant tokens
# meanp: EEG models may not use a [cls] token, so use mean of tokens as a replacement of [cls]. This approach depends on the cornve space
# clstrpts (TR-PTS, a merging version of [cls] pruning): 
# meantrpts (TR-PTS, a merging version of mean pruning): 
methods=(
    tome 
    tofu 
    clsp 
    meanp 
    clsevit2
    meanevit2
    'dartpo[k,x]'
) 

ds=TUAB
for m in "${methods[@]}"; do
    for r in 1 4 8 12 15 19; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 19; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=TUEV
for m in "${methods[@]}"; do
    for r in 1 2 4 6 8 9; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 9; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=ISRUC
for m in "${methods[@]}"; do
    for r in 1 3 6 9 12 14; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 14; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=EEGMAT
for m in "${methods[@]}"; do
    for r in 1 2 3 4 5; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 5; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=EarEEG
for m in "${methods[@]}"; do
    for r in 1 2 4 6 8 9; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    # for r in 9; do
    #     python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    # done
done
