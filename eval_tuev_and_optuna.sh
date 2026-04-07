#!/bin/bash

python run_infer_multiclass.py --dataset TUEV --no_train --test --workspace inference_tuev --log_dir baselines --model BIOT
python run_infer_multiclass.py --dataset TUEV --no_train --test --workspace inference_tuev --log_dir baselines --model LaBraM
python run_infer_multiclass.py --dataset TUEV --no_train --test --workspace inference_tuev --log_dir baselines --model TFM

for r in 1 7 14 22 19; do
    python run_infer_multiclass.py --dataset TUEV --no_train --test --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model BIOT
    python run_infer_multiclass.py --dataset TUEV --no_train --test --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model TFM

    python optuna_multiclass.py --dataset TUEV --test --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model BIOT
    python optuna_multiclass.py --dataset TUEV --test --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model TFM
done

for r in 1 2 4 6 8 9; do
    python run_infer_multiclass.py --dataset TUEV --no_train --test --tome_variant tome[q] --tome_r $r --workspace inference_tuev --log_dir tome --model LaBraM
    python optuna_multiclass.py --dataset TUEV --test --tome-scheme --tome_variant kiddp[q] --tome_r $r --storage sqlite:///optuna_tuev.sqlite3 --model LaBraM
done

