#!/bin/bash

common_args=(
    run_inference.py
    --num_workers 8
    --test
    --batch_size ${1:-'128'}
    --model TFM
    --workspace inference_tfm
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
    # 'dartp_[k,x]'
    'dartpo[k,x]'
) 

ds=TUAB
for m in "${methods[@]}"; do
    # for r in 1 19 38 57 61 75; do # 1/4
    for r in 15 30 46 61 75; do # 1/5
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 61 75; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=TUEV
for m in "${methods[@]}"; do
    # for r in 1 9 18 27 29 35; do # 1/4
    for r in 7 14 22 29 35; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 29 35; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=ISRUC
for m in "${methods[@]}"; do
    for r in 1 18 35 53 71 88; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 71 88; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=EEGMAT
for m in "${methods[@]}"; do
    for r in 1 6 11 17 22 27; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 27; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done

ds=EarEEG
for m in "${methods[@]}"; do
    for r in 1 12 24 35 47 58; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds} --tome-scheme
    done
    for r in 58; do
        python $common_args --dataset $ds --tome_variant $m --tome_r $r --log_dir ${m}-${ds}
    done
done
