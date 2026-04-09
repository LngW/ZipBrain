#!/bin/bash

common_args=(
    --num_workers 8 
    --dataset TUEV
    --test 
    --batch_size 128 
    --workspace optuna_tuev
    --n_trials_qmc 128
    --n_trials_tpe 0
    --storage sqlite:///tuev.sqlite3
)
common_args="${common_args[@]}"
fname=run_optuna_multiclass.py
ts=--tome-scheme

# ToMe, ToFU, clsp, meanp, meanm0
for r in 1 9 18 27 29 35; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model BIOT
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model BIOT
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model BIOT
done
for r in 1 2 4 6 8 9; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model LaBraM
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model LaBraM
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model LaBraM
done
for r in 1 9 18 27 29 35; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model TFM
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model TFM
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model TFM
done

# Non Tome Scheme, only a partial of experiments should be re-done
ts=''
for r in 29 35; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model BIOT
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model BIOT
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model BIOT
done
for r in 9; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model LaBraM
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model LaBraM
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model LaBraM
done
for r in 29 35; do
    python $fname $common_args $ts --tome_variant kiddp --tome_r $r --log_dir kiddp --model TFM
    python $fname $common_args $ts --tome_variant kiddl --tome_r $r --log_dir kiddl --model TFM
    python $fname $common_args $ts --tome_variant kiddl2 --tome_r $r --log_dir kiddl2 --model TFM
done
