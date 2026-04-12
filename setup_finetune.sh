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

if [ ! -e finetune/TFM_Tokenizer/ ]; then
    echo Copying TFM-Tokenizer from ./thirdparty/ to ./finetune/
    cp -r thirdparty/TFM_Tokenizer/ finetune/
fi

apply_patch()
{
    patch thirdparty/$1 -i finetune/patches/$1.patch -o finetune/$1
}

apply_patch LaBraM/run_class_finetuning.py
apply_patch LaBraM/engine_for_finetuning.py
apply_patch LaBraM/utils.py
apply_patch BIOT/run_multiclass_supervised.py
apply_patch BIOT/run_binary_supervised.py
apply_patch TFM_Tokenizer/downstream_transformer_finetuning.py

cur_dir=$(pwd)

ln -s $(pwd)/data_loaders.py finetune/BIOT/data_loaders.py
ln -s $(pwd)/data_loaders.py finetune/LaBraM/data_loaders.py
ln -s $(pwd)/data_loaders.py finetune/TFM_Tokenizer/data_loaders.py

ln -s $(pwd)/dataset_configs.finetune.yaml finetune/BIOT/dataset_configs.yaml
ln -s $(pwd)/dataset_configs.finetune.yaml finetune/LaBraM/dataset_configs.yaml
ln -s $(pwd)/dataset_configs.finetune.yaml finetune/TFM_Tokenizer/dataset_configs.yaml

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

echo
echo setting up venv for TFM_Tokenizer
cd $cur_dir/finetune/TFM_Tokenizer
uv venv -p 312 --clear
uv pip sync -q ../requirements.tfm.txt --torch-backend cu118

cd $cur_dir

echo
echo Environments setup finished