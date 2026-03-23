#!/bin/bash

common_args=(
    --dataset TUAB
    --no_train
    --test
    --workspace inferences
)

# for model in 'BIOT' 'TFM'; do
for r in 19 38 54 "75 75 75 39" 75; do
    for m in 'tome[q]' 'kiddp[q]'; do
        python run_infer_binary.py --model BIOT --tome_r $r --tome_variant $m "${common_args[@]}" --log_dir BIOT
    done
done
for r in 19 38 54 "75 75 75 39" 75; do
    for m in 'tome[q]' 'kiddp[q]'; do
        PYTHONPATH="$PYTHONPATH:./thirdparty/TFM_Tokenizer/" python run_infer_binary.py --model TFM --tome_r $r --tome_variant $m "${common_args[@]}" --log_dir TFM
    done
done

# done