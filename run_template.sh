#!/bin/bash
tag="test_1"
args="--epochs 2 --subset 10 --batch_size 128 --lr 1e-4 --num_workers 4"
# models="--model BIOT T9BIOT T19BIOT SABIOT ToMeBIOTr38"
models="--model BIOT"
# used seeds are : 12345856, 85875035, 46812486, 68486431, 86435434, 34525135
python experiments.py $args $models --tag $tag --seed 12345865

