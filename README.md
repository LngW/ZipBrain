## How to use this repository

### 1. Clone our repository

```shell
git clone https://github.com/LngW/ZipBrain
git submodule update --init
```

We use git's `submodule` to manage upstream repositories, so you don't have to clone them by youself. You can easily fetch and clone those repositories by using `git submodule update --init`.

Please note that we use files from upstreams for finetuning and model defination, so 
please **MAKE SURE** that you initialized and fetched upstreams. You would see folders 
named `BIOT`, `LaBraM`, `TFM_Tokenizer`, `EEGPT`, `ToMe` and so on in `./thirdparty/`.

> In case the repository is downloaded as zip file, you can add submodules manually.
> Please refer to the [.gitmodules](.gitmodules) for links

### 2. Finetune upstream models

Token Compression methods are adopted to finetuned models, so a finetuned checkpoint is 
necessary to evaluate our method.

We provide a simple shell script to setup finetuning environments
```shell
sh setup_finetune.sh
```

After running the command, folders named `BIOT`, `LaBraM` and `TFM_Tokenizer` can be found under `./finetune/`. Please then follow instructions from those upstreams to obtain a finetuned checkpoint.

We provide shell scripts (`./finetune/finetune.<model>.sh`) that record hyper-parameters we used in finetuning upstream models. Readers can use these scripts for faster replicate of our results. 

### 3. Evaluate our method and baselines

[`run_inference.py`](run_inference.py) defines a unified entrance for evaluating all baselines and our method.

You can use the following command to evaluate our method
```sh
python run_inference.py --model BIOT --dataset TUEV --tome_variant 'kiddl1s[k,k,k]' --tome_r 35 --pivot-factor 0.45 --imp-factor 1 --test --workspace evaluate_biot --log_dir evaluate_biot
```

To evaluate other token compression methods, please replace 'kiddl1s[k,k,k]' with other variant names.  
Currently we support:
* ToMe: `--tome_variant tome[k]`
* ToFU: `--tome_variant tofu[k]`
* EViT: `--tome_variant clsevit2`
* EViT(mean): `--tome_variant meanevit2`
* DART: `--tome_variant 'dartpo[k,x]'`

For Datasets, we support:
* TUAB: `--dataset TUAB`
* TUEV: `--dataset TUEV`
* ISRUC: `--dataset ISRUC`
* EEGMAT/WORKLOAD: `--dataset EEGMAT`
* EarEEG: `--dataset EerEEG`

> A warning about TUAB and TUEV  
> 
> For TUAB and TUEV, we used pre-processing pipelines defined in BIOT, LaBraM and TFM-Tokenizer for the three models seperately.

> A warning about ISRUC
> 
> The dataset is reformed with lmdb and lz4 for faster loading speed.

For Models, we support:
* BIOT: `--model BIOT`
* LaBraM: `--model LaBraM`
* TFM-Tokenizer: `--model TFM`
* EEGPT: `--model EEGPT`

### 4. Optimal Hyper-Parameter
File `run_optuna_grid_full_m3.py` and `run_optuna_grid_copy_study.py` defines the hyper-parameter optimization pipeline. 

We provide scripts `run_optuna_<model>_m3.sh` for faster replicate.
