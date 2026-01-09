#!/bin/bash
args="--epochs 2 --subset 10 --batch_size 128 --lr 1e-4 --num_workers 4"
seeds="--seed 12345856 85875035 46812486 68486431 86435434 34525135"
# models="--model BIOT T9BIOT T19BIOT SABIOT ToMeBIOTr38"
models="--model ToMeBIOTr38"
python experiments.py $args $seeds $models --tag test_1

