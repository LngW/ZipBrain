#!/bin/bash
# workspace=swag_tome
common_args=(
    --seed 0
    --test
    --workspace swag_test
)

if ! [ -d './workspace' ]; then
    mkdir ./workspace
fi

run_test()
{
    variant=$1
    r=$2

    python run_binary_supervised2.py "${common_args[@]}" --tome_variant $variant --log_dir ${variant}_${r} --tome_r $r
}

for r in 0 10 20 30 40 50 60 70 80 90 100; do
    run_test tome $r
done

for r in 0 10 20 30 40 50 60 70 80 90 100; do
    run_test f $r
done