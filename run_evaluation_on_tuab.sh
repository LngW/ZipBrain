#!/bin/bash

common_args="--num_workers 8 --dataset TUAB --test --tome-scheme --batch_size 128 --workspace inference_tuab"
fname=run_infer_binary.py

# Baselines
python $fname $common_args --log_dir baselines --model BIOT
python $fname $common_args --log_dir baselines --model LaBraM
python $fname $common_args --log_dir baselines --model TFM

# ToMe, ToFU, clsp, meanp, meanm0
for r in 1 19 38 57 61 75; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model BIOT
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model BIOT
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model BIOT
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model BIOT
done
for r in 1 4 8 12 15 19; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model LaBraM
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model LaBraM
    python $fname $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model LaBraM
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model LaBraM
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model LaBraM
done
for r in 1 19 38 57 61 75; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model TFM
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model TFM
    python $fname $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model TFM
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model TFM
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model TFM
done

# Non Tome Scheme, only a partial of experiments should be re-done
common_args="--num_workers 8 --dataset TUAB --test --batch_size 128 --workspace inference_tuab"

for r in 61 75; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model BIOT
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model BIOT
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model BIOT
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model BIOT
done
for r in 19; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model LaBraM
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model LaBraM
    python $fname $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model LaBraM
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model LaBraM
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model LaBraM
done
for r in 61 75; do
    python $fname $common_args --tome_variant tome --tome_r $r --log_dir tome --model TFM
    python $fname $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model TFM
    python $fname $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model TFM
    python $fname $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model TFM
    python $fname $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model TFM
done
