#!/bin/bash
# workspace=swag_tome
common_args=(
    --seed 0
    --workspace swag_384_tome_r_scan
    # --debug
    # --valid
    --test
    --model b16_swag
    --dataset ImageNetWoof
    --batch_size 64
)

if ! [ -d './workspace' ]; then
    mkdir ./workspace
fi

rfile=run_infer_swag_imwf.py
# rfile="-m debugpy --listen 6789 --wait-for-client $rfile"

function run_test()
{
    variant=$1
    r=$2
    t=($2)
    python $rfile "${common_args[@]}" --tome_variant $variant --log_dir ${variant}-cls-$t --tome_r $r
}

# for r in 0 10 20 30 40 50 60 70 80 90 100; do
#     run_test tome $r
# done

# for r in 0 10 20 30 40 50 60 70 80 90 100; do
#     run_test f $r
# done

# run_test f 47
# exit

# for m in tome f fx; do
#     for r in 0 5 11 17 23 29 35 41 47; do
#         run_test $m $r
#     done
# done
# for m in tome f fx fh fhx fs fsx; do
#     for r in 0 5 11 17 23 29 35 41 47; do
#         run_test $m $r
#     done
# done
# run_test mean 0 5 11 17 23 29 35 41 47
# for m in tome mean; do
#     for r in 0 5 11 17 23 29 35 41 47; do
#         # run_test tome $r
#         run_test $m $r
#     done
# done

# for m in 'kiddp[k]' 'dartp[k]' 'dartm[k]' 'tome[k]'; do
# for r in 5 17 29 41; do
#     # run_test 'tome[q]' $r 0
#     # run_test 'tome[k]' $r 0
#     for pf in 0.1 0.2 0.4 0.8; do
#         # run_test 'dartp[q]' $r $pf
#         run_test 'dartp[k]' $r $pf
#     done
# done

# exit

for m in 'kiddp[k]'; do
    for r in {1..143}; do
        run_test tome "$r 0 0 $r 0 0 $r 0 0 $r 0 0"
    done
done