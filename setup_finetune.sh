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

if [ ! -e finetune/EEGPT/ ]; then
    echo Copying EEGPT from ./thirdparty/ to ./finetune/
    cp -r thirdparty/EEGPT/ finetune/
fi

apply_patch()
{   
    local tar=$1
    local src=${2:-"$tar"}
    local base=${3:-"thirdparty"}
    patch $base/$src -i finetune/patches/$tar.patch -o finetune/$tar
}

apply_patch LaBraM/run_class_finetuning.py
apply_patch LaBraM/engine_for_finetuning.py
apply_patch LaBraM/utils.py

apply_patch BIOT/run_multiclass_supervised.py
apply_patch BIOT/run_binary_supervised.py

apply_patch TFM_Tokenizer/downstream_transformer_finetuning.py

apply_patch EEGPT/downstream_tueg/finetune_TUAB_EEGPT.sh
apply_patch EEGPT/downstream_tueg/finetune_TUEV_EEGPT.sh
apply_patch EEGPT/downstream_tueg/engine_for_finetuning_EEGPT.py
apply_patch EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py
apply_patch EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
apply_patch EEGPT/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py
apply_patch EEGPT/downstream_tueg/utils.py

apply_patch EEGPT/downstream/finetune_EEGPT_EarEEG.py EEGPT/downstream/finetune_EEGPT_SleepEDF.py
apply_patch EEGPT/downstream/finetune_EEGPT_EEGMAT.py EEGPT/downstream/finetune_EEGPT_SleepEDF.py
apply_patch EEGPT/downstream/finetune_EEGPT_ISRUC.py EEGPT/downstream/finetune_EEGPT_SleepEDF.py

apply_patch EEGPT/downstream/dataset_configs.yaml dataset_configs.yaml .

cur_dir=$(pwd)

ln -s $(pwd)/data_loaders.py finetune/BIOT/data_loaders.py
ln -s $(pwd)/data_loaders.py finetune/LaBraM/data_loaders.py
ln -s $(pwd)/data_loaders.py finetune/TFM_Tokenizer/data_loaders.py

ln -s $(pwd)/dataset_configs.finetune.yaml finetune/BIOT/dataset_configs.yaml
ln -s $(pwd)/dataset_configs.finetune.yaml finetune/LaBraM/dataset_configs.yaml
ln -s $(pwd)/dataset_configs.finetune.yaml finetune/TFM_Tokenizer/dataset_configs.yaml

ln -sT $cur_dir/datasets/tuab/labram datasets/tuab/eegpt
ln -sT $cur_dir/datasets/tuev/labram datasets/tuev/eegpt

echo
echo setting up venv for BIOT
cd $cur_dir/finetune/BIOT/
if [ ! -e .venv ]; then
    uv venv -p 312 --clear
fi
uv pip sync -q ../requirements.biot.txt --torch-backend cu118

echo
echo setting up venv for LaBraM
cd $cur_dir/finetune/LaBraM
if [ ! -e .venv ]; then
    uv venv -p 312 --clear
fi
uv pip sync -q ../requirements.labram.txt --torch-backend cu118

echo
echo setting up venv for TFM_Tokenizer
cd $cur_dir/finetune/TFM_Tokenizer
if [ ! -e .venv ]; then
    uv venv -p 312 --clear
fi
uv pip sync -q ../requirements.tfm.txt --torch-backend cu118

echo
echo setting up venv for EEGPT
cd $cur_dir/finetune/EEGPT
if [ ! -e .venv ]; then
    uv venv -p 312 --clear
fi
uv pip sync -q ../requirements.eegpt.txt --torch-backend cu118

cd $cur_dir

echo
echo Environments setup finished