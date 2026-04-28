## How to use

### 1. Clone our repository and upstream repositories

```shell
git clone https://github/LngW/Kidd
git submodule update --init
```

We use submodule for upstream repositories, so you don't have to clone them by youself.

Please note that we used files from upstreams for finetuning and also model defination, so 
please **MAKE SURE** you initialized and updated submodules. You would see folders 
named BIOT, LaBraM, TFM_Tokenizer, ToMe and so on in `./thirdparty/`.

### 2. Finetune upstream models

Token Compression methods are adopted to finetuned models, so a finetuned checkpoint is 
necessary to evaluate our method.

Please follow instructions from upstreams to obtain a finetuned checkpoint.

We provide a simple shell script to setup finetuning environments
```shell
sh setup_finetune.sh
```

After running the command, folders named `BIOT`, `LaBraM` and `TFM_Tokenizer` can be found under `./finetune/`.

For your easy replication of our results, we also provide dataset splittion and commandline to obtain our checkpoint.

### 3. Evaluate our method and baselines

[run_inference.py](run_inference.py) defines a unified entrance for evaluating all baselines and our method.

You can use the following command to evaluate our method
```sh
python run_inference.py --model BIOT --dataset TUEV --tome_variant 'kiddl2[k]' --tome_r 35 --pivot-factor 0.45 --imp-factor 0.6 --test --workspace evaluate_biot --log_dir evaluate_biot
```

To evaluate other token compression methods, please replace 'kiddl2[k]' with other varaint names.  
Currently we support:
* ToMe: `--tome_variant tome`
* ToFU: `--tome_variant tofu`
* FasterVLM: `--tome_variant clsp`
* TR-PTS: `--tome_variant clsm`
* FasterVLM*: `--tome_variant meanp`
* TR-PTS*: `--tome_variant meanm`
* DART^: `--tome_variant 'dartp[k,x]'`

## Integrate into new models

For easier intergration into other models, we introduce here about how to integrate our method into other models.

Our method is inserted between attention mechanism and FFN part and uses K, Q, V matrices generated in attention mechanism as inputs.

The following pseudo python code demonstrate how to use our method.

```
# x1 is of shape (bsz, seq, dim)
# k, q, v are of shape (bsz, H, seq, HD)
# To use our method, the implementation of Attention should be modified, to collect k,q,v matrices.
x1, k, q, v = Attention(x0) 

# Here x2 is of shape (bsz, max(seq - r, 1), dim)
x2 = apply_merge(pinfo, r, 'kiddl2[k]', x1, q, k, v)

# Nothing changed about and after this line of code.
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

### **The Patching Way**

When you want to inject code without modifying the original source files, you can use Python’s **duck typing** feature. You can find an example implementation in [`./patch/biot.py#L116`](./patch/biot.py#L116).

#### **1. Identify Target Classes**
First, identify which classes need to be patched. We recommend patching at least the following:
* **The Main Model Class:** To set initialization parameters and reset computation caches.
* **The Attention Class:** To collect $Q, K, V$ matrices.
* **The Transformer Block Class (which calls Attention and FFN):** To insert our custom logic into the execution flow.

#### **2. Implementation Details**
* **Main Class:** In [`biot.py`](./patch/biot.py), we modified the [`BIOTClassifier`](./thirdparty/BIOT/model/biot.py#L148) class by [`PatchedClassifier`](./patch/biot.py#L8) class to include parameter initialization.
* **Attention & FFN:** The BIOT model uses [`LinearAttentionTransformer`](https://github.com/lucidrains/linear-attention-transformer/tree/0.19.1). We defined patched versions of the [`SequentialSequence`](https://github.com/lucidrains/linear-attention-transformer/blob/0.19.1/linear_attention_transformer/reversible.py#L133) and [`LinformerSelfAttention`](https://github.com/lucidrains/linformer/blob/0.2.3/linformer/linformer.py#L66) classes as [`PatchedSequentialSequence`](./patch/linear_attn_transformer.py#L26) and [`PatechedSelfAttention`](./patch/linear_attn_transformer.py#L54) in [`linear_attn_transformer.py`](./patch/linear_attn_transformer.py).
* **Data Handling:** To avoid cluttering method signatures or return values, we use the configuration dict `pinfo` as a central storage for $Q, K, V$ matrices and other metadata like `size` or `source`.

#### **3. Calling the Patched Model**

Once you have implemented the patching method, register it in [`./patch/__init__.py`](./patch/__init__.py). You can then apply the patch to your model as follows:

```python
# 1. Create instance and load checkpoints
model = prepare_model() 

# 2. Apply the patch using your registered model name
patch.your_model_name(
    model,
    trace_source = False,
    show_shape = False,
    tome_scheme = False
)

# 3. Configure hyperparameters
model.tome_variant = 'kiddl2[k]'
model.tome_r = 1
model.pivot_factor = 0.45
model.imp_factor = 0.6

# 4. Use the model as usual
pred = model(x) 
```

Notice that, in this case, the `pinfo` should be generated when applying patches to the model, as done in [`apply_patch`](./patch/biot.py#L137-L149) and [`PatchedClassifier`](./patch/biot.py#L13-L25).
