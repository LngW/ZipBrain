#!/bin/bash

if [ ! -e thirdparty/BIOT/ ]; then
    echo Copying BIOT from ./thirdparty/ to ./finetune/
    cp -r thirdparty/BIOT/ finetune/
fi

if [ ! -e thirdparty/LaBraM/ ]; then
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
cd $cur_dir/finetune/BIOT/
uv venv -p 312 --clear
uv pip sync ../requirements.biot.txt --torch-backend cu118

cd $cur_dir/finetune/LaBraM
uv venv -p 312 --clear
uv pip sync ../requirements.labram.txt --torch-backend cu118

cd $cur_dir