#!/bin/bash

common_args=(
    --dataset TUAB
    --no_train
    --test
    --workspace inferences
    # --batch_size 128
    --batch_size 64
)

export CUDA_VISIBLE_DEVICES=0

# for model in 'BIOT' 'TFM'; do
# for r in 0 19 38 54 "75 75 75 39" 75; do
#     for m in 'tome[q]' 'kiddp[q]'; do
#         python run_infer_binary.py --model BIOT --tome_r $r --tome_variant $m "${common_args[@]}" --log_dir BIOT
#     done
# done
# for r in 0 19 38 54 "75 75 75 39" 75; do
#     for m in 'tome[q]' 'kiddp[q]'; do
#         PYTHONPATH="$PYTHONPATH:./thirdparty/TFM_Tokenizer/" python run_infer_binary.py --model TFM --tome_r $r --tome_variant $m "${common_args[@]}" --log_dir TFM
#     done
# done
for r in 0 3 7 11 15 19; do
    for m in 'tome[k]' 'kiddp[k]'; do
        PYTHONPATH=$PYTHONPATH:./thirdparty/LaBraM/ python run_infer_binary.py --model LaBraM --dataset TUAB --tome_r $r --tome_variant $m --no_train --test --workspace inferences --log_dir LaBraM
    done
done
# done