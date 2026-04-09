#!/bin/bash

common_args="--num_workers 8 --dataset TUEV --test --tome-scheme --batch_size 512 --workspace inference_tuev"

# Baselines
python run_infer_multiclass.py $common_args --log_dir baselines --model BIOT
python run_infer_multiclass.py $common_args --log_dir baselines --model LaBraM
python run_infer_multiclass.py $common_args --log_dir baselines --model TFM

# ToMe, ToFU, clsp, meanp, meanm0
for r in 1 9 18 27 29 35; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model BIOT
done
for r in 1 2 4 6 8 9; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model LaBraM
done
for r in 1 9 18 27 29 35; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model TFM
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model TFM
    python run_infer_multiclass.py $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model TFM
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model TFM
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model TFM
done

# Non Tome Scheme, only a partial of experiments should be re-done
common_args="--num_workers 8 --dataset TUEV --test --batch_size 512 --workspace inference_tuev"

for r in 29 35; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model BIOT
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model BIOT
done
for r in 9; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model LaBraM
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model LaBraM
done
for r in 29 35; do
    python run_infer_multiclass.py $common_args --tome_variant tome --tome_r $r --log_dir tome --model TFM
    python run_infer_multiclass.py $common_args --tome_variant tofu --tome_r $r --log_dir tofu --model TFM
    python run_infer_multiclass.py $common_args --tome_variant clsp --tome_r $r --log_dir clsp --model TFM
    python run_infer_multiclass.py $common_args --tome_variant meanp --tome_r $r --log_dir meanp --model TFM
    python run_infer_multiclass.py $common_args --tome_variant meanm0 --tome_r $r --log_dir meanm0 --model TFM
done
