## How to use

### Clone our repository and upstream repositories

```shell
git clone https://github/LngW/Kidd
git submodule update --init
```

We use submodule for upstream repositories, so you don't have to clone them by youself.

Please note that we used files from upstreams for finetuning and also model defination, so 
please **MAKE SURE** you initialized and updated submodules. You would see folders 
named BIOT, LaBraM, TFM_Tokenizer, ToMe and so on in ./thirdparty/.

### Finetune upstream models

Token Compression methods are adopted to finetuned models, so a finetuned checkpoint is 
necessary to evaluate our method.

Please follow instructions from upstreams to obtain a finetuned checkpoint.

We provide a simple script to setup finetuning environment
```shell
sh setup_finetune.sh
```

For your easy replication of our results, we also provide dataset splittion and commandline to obtain our checkpoint.

### Evaluate our method and baselines

[run_inference.py](run_inference.py) defines a unified entrance for evaluating all baselines and our method.

You can use the following command to evaluate our method
```sh
python run_inference.py --model BIOT --dataset TUEV --tome_variant 'kiddl2[k]' --tome_r 35 --pivot-factor 0.45 --imp-factor 0.6 --test --workspace evaluate_biot --log_dir evaluate_biot
```

To evaluate other token compression methods, please replace 'kiddl2[k]' with other varaint names.  
Currently we support:
* ToMe: tome
* ToFU: tofu
* FasterVLM: clsp
* TR-PTS: clsm
* FasterVLM*: meanp
* TR-PTS*: meanm
* DART^: 'dartp[k,x]'

## Integrate into new models

For easier intergration into other models, we introduce here about how to integrate our method into other models.

Our method is inserted between attention mechanism and FFN part and uses K, Q, V matrices generated in attention mechanism as inputs.

The following pseudo python code demonstrate how our method works.


```
x1, k, q, v = Attention(x0) # Here x1 is of shape (bsz, seq, dim) and k, q, v are of shape (bsz, H, seq, HD)
x2 = apply_merge(pinfo, r, 'kiddl2[k]', x1, q, k, v) # Here x2 is of shape (bsz, max(seq - r, 1), dim)
x3 = FFN(x2)
```

The function `apply_merge` is defined in [merge_variants](./merges/merge_variants.py#L18) and `pinfo` here is a dict of configurations.

Specially for our method kiddl2, the following config can be used as a default config.

```python
pinfo = {
    'class_token': False, # This value should be set to true if the model uses [cls] token for classification
    'distill_token': False,
    'trace_source': False,
    'tome_shceme': False,
    'use_cls': False,
    'pivot_factor': [0.45],
    'imp_factor': [0.6],
}
```

Both `pivot_factor` and `imp_factor` affect the performance of our method heavily so they should be adjusted carefully.