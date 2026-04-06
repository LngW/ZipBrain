#!/bin/bash

if ! command -v uv >/dev/null 2>&1; then
    echo We use uv to manage python envs, please install it to use this script.
    echo You can donwload it from: https://docs.astral.sh/uv/
    exit
fi

if [ ! -e finetune/BIOT/ ]; then
    echo Copying BIOT from ./thirdparty/ to ./finetune/
    cp -r thirdparty/BIOT/ finetune/
fi

if [ ! -e finetune/LaBraM/ ]; then
    echo Copying LaBraM from ./thirdparty/ to ./finetune/
    cp -r thirdparty/LaBraM/ finetune/
fi

apply_patch()
{
    patch thirdparty/$1 -i finetune/patches/$1.patch -o finetune/$1
}

apply_patch LaBraM/run_class_finetuning.py
apply_patch LaBraM/engine_for_finetuning.py
apply_patch BIOT/run_multiclass_supervised.py
apply_patch BIOT/run_binary_supervised.py

cur_dir=$(pwd)

echo
echo setting up venv for BIOT
cd $cur_dir/finetune/BIOT/
uv venv -p 312 --clear
uv pip sync -q ../requirements.biot.txt --torch-backend cu118

echo
echo setting up venv for LaBraM
cd $cur_dir/finetune/LaBraM
uv venv -p 312 --clear
uv pip sync -q ../requirements.labram.txt --torch-backend cu118

cd $cur_dir

echo
echo Environments setup finished