#!/bin/bash

common_args="--num_workers 8 --dataset TUEV --test --n_trials_qmc 128 --n_trials_tpe 0 --suffix mo-128-0"

# python run_infer_multiclass.py $common_args --no_train --workspace inference_tuev --log_dir baselines --model BIOT
# python run_infer_multiclass.py $common_args --no_train --workspace inference_tuev --log_dir baselines --model LaBraM
# python run_infer_multiclass.py $common_args --no_train --workspace inference_tuev --log_dir baselines --model TFM

for r in 1 7 14 22 19; do
    # python run_infer_multiclass.py $common_args --no_train --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model BIOT
    # python run_infer_multiclass.py $common_args --no_train --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model TFM

    python optuna_multiclass.py $common_args --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model BIOT
    python optuna_multiclass.py $common_args --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model TFM
done

for r in 1 2 4 6 8 9; do
    # python run_infer_multiclass.py $common_args --no_train --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model LaBraM
    python optuna_multiclass.py $common_args --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model LaBraM
done

