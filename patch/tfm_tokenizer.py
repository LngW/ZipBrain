
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

def make_classifier_class(klass):
    import torch
    from einops import rearrange

    class PatchedTFMClassifier(klass):
        def forward(self, x,num_ch = 16):

            x = self.eeg_token_embedding(x)

            for i in range(x.shape[1]):
                used_channel_embed = self.channel_embed(self.index[i]).unsqueeze(0).unsqueeze(0).expand(x.size(0),-1,-1)
                x[:,i] = self.temporal_pos_embed(x[:,i]+used_channel_embed)
            
            ch_emb = self.channel_embed(self.index[0:x.shape[1]]).unsqueeze(1).unsqueeze(0)
            ts_emb = self.temporal_pos_embed.pe[:, :x.shape[2]].unsqueeze(1)
            self._pinfo['pte'] = (ch_emb + ts_emb).flatten(1, 2).expand(x.shape[0], -1, -1)

            x = rearrange(x, 'B C T E -> B (C T) E')

            
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            x = self.LAT(x)
            pred = self.classification_head(x[:, 0])
            return pred
        
    return PatchedTFMClassifier

def apply_patch(model, trace_source = False, show_shape = False, tome_scheme = False):
    if model.__class__.__name__ != 'Pl_tfm_tokenizer_inference':
        return
    
    transformer = model.tfm_token.LAT
    from .linear_attn_transformer import make_sequential_class, make_self_attention_class
    # from .biot import make_sequential_class, make_self_attention_class
    PatchedInference = make_PL_class(model.__class__)
    PatchedClassifier = make_classifier_class(model.tfm_token.__class__)
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
        "show_shape": show_shape,
        "tome_scheme": tome_scheme,
    }

    model.__class__ = PatchedInference
    model.r = 0
    model.variant = ''
    model._pinfo = pinfo
    model.pivot_factor = 0.05
    model.use_cls = True

    model.tfm_token.__class__ = PatchedClassifier
    model.tfm_token._pinfo = pinfo

    lat = model.tfm_token.LAT
    lat.layers.__class__ = PatchedSequence
    lat.layers._pinfo = pinfo

    for it in lat.layers.layers:
        it[0].fn.__class__ = PatchedSelfAttention
        it[0].fn._pinfo = pinfo

    print('Patched TFM-Tokenizer, Layers = {}'.format(len(lat.layers.layers)))

if __name__ == '__main__':

    import sys
    import importlib

    sys.modules['utils'] = importlib.import_module('thirdparty.TFM_Tokenizer.utils')
    sys.modules['datasets.data_loaders'] = importlib.import_module('thirdparty.TFM_Tokenizer.datasets.data_loaders')
    sys.modules['models.tfm_token'] = importlib.import_module('thirdparty.TFM_Tokenizer.models.tfm_token')

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
