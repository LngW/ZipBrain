#!/bin/bash

common_args=(
    # --seed 1
    --model BIOT
    --dataset TUAB
    --subset 100
    --test
    --no_train
    # --valid
    # --debug
    --num_workers 4
    --workspace inference_tuab100_kidd_compares
)

rfile=run_binary_inference.py
# rfile="-m debugpy --listen 6789 --wait-for-client $rfile"

function run()
{   
    local m="$1[$2]"
    local r="$3"

    if [ "$#" -gt 3 ]; then
        local log_dir="$4-${r// /_}"
    else
        local log_dir="${m}-${r// /_}"
    fi

    # echo $rfile ${common_args[@]} --tome_r $r --tome_variant $m --log_dir $log_dir
    python $rfile ${common_args[@]} --tome_r $r --tome_variant $m --log_dir $log_dir
}

for r in 0 19 38 57 "75 75 75 39" 75; do
    run kiddp q "$r" kidd_pivot
    run kiddl q "$r" kidd_left
    run tome q "$r"
    run dartp q "$r"
    run dartm q "$r"

    run "dartp dartp dartm dartm" q "$r" 2dartp2dartm
done
exit

