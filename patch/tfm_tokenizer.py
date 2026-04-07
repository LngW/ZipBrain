
from typing import NamedTuple

from .utils import reset_common_pinfo

def make_PL_class(klass):
    class PatchInference(klass):
        def forward(self, *args, **kwargs):
            depth = len(self.tfm_token.LAT.layers.layers)
            pinfo = self._pinfo
            reset_common_pinfo(self, pinfo, depth)
            # pinfo["r"] = parse_r(depth, self.r)
            # pinfo["variant"] = parse_variant(depth, self.variant)
            pinfo['shape'] = None
            pinfo["size"] = None
            pinfo["source"] = None
            pinfo["qkv"] = None

            # self._pinfo["pivot_factor"] = self.pivot_factor
            # if hasattr(self, 'pivot_const'):
            #     self._pinfo['pivot_const'] = self.pivot_const
            self._pinfo["use_cls"] = self.use_cls
            # self._pinfo["pe_score"] = None
            # self._pinfo["alibi"] = None
            # self._pinfo["attn_score"] = None

            return super().forward(*args, **kwargs)

    return PatchInference

def apply_patch(model, trace_source = False, show_shape = False):
    if model.__class__.__name__ != 'Pl_tfm_tokenizer_inference':
        return
    
    transformer = model.tfm_token.LAT
    from .linear_attn_transformer import make_sequential_class, make_self_attention_class
    # from .biot import make_sequential_class, make_self_attention_class
    PatchedInference = make_PL_class(model.__class__)
    PatchedSequence = make_sequential_class(transformer.layers.__class__)
    # print(transformer.layers.layers[0][0])
    PatchedSelfAttention = make_self_attention_class(transformer.layers.layers[0][0].fn.__class__)

    pinfo = {
        'r': 0,
        'variant': '',
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": False,
        "class_token": True,
        "distill_token": False,
        "show_shape": show_shape
    }

    model.__class__ = PatchedInference
    model.r = 0
    model.variant = ''
    model._pinfo = pinfo
    model.pivot_factor = 0.05
    model.use_cls = True

    lat = model.tfm_token.LAT
    lat.layers.__class__ = PatchedSequence
    lat.layers._pinfo = pinfo

    for it in lat.layers.layers:
        it[0].fn.__class__ = PatchedSelfAttention
        it[0].fn._pinfo = pinfo

    print('Patched TFM-Tokenizer, Layers = {}'.format(len(lat.layers.layers)))

if __name__ == '__main__':
    from thirdparty.TFM_Tokenizer.tfm_tokenizer_inference import Pl_tfm_tokenizer_inference
    # import yaml
    # import argparse
    # import os
    from types import SimpleNamespace
    import torch



    args = SimpleNamespace()
    # 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth'
    args.vqvae_pretrained_path = 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth'
    args.code_book_size = 8192
    args.emb_size = 64
    args.finetuned_path = 'thirdparty/TFM_Tokenizer/pretrained_weigths/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth'
    args.resampling_rate = 200

    dataset_params = {
        'classification_task': 'binary',
        'num_classes': 1
    }

    model = Pl_tfm_tokenizer_inference(
        args = args,
        training_params = None,
        save_path = './workspace/debug/debug/',
        niter_per_ep = 1,
        dataset_params = dataset_params
    )

    print(len(model.tfm_token.LAT.layers.layers))

    x = torch.rand(1, 16, 2000)
    print(model(x))

    apply_patch(model)
    print(model(x))

    model.r = 1
    model.variant = 'tome'
    print(model(x))
