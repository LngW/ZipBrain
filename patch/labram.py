
from .utils import reset_common_pinfo, compute_cinfo

def apply_merge(pinfo, x, q, k, v):

    # q, k, v = pinfo['qkv']
    # pinfo['qkv'] = None

    # if r <= 0:
    #     return x

    # metrics = {'q': q, 'k': k, 'v': v, 'x': x}
    # def _find_metric(variant):
    #     left = variant.find('[')
    #     right = variant.find(']')
    #     if left > 0 and right > left:
    #         return fast_dict.get(variant[left + 1 : right], k)
    #     else:
    #         return k
    
    # if variant is not None:
    #     metric = _find_metric(variant)

    cinfo = compute_cinfo(pinfo)

    from merges import apply_merge_impl
    return apply_merge_impl(pinfo, cinfo, x, q, k, v)


def create_block_class(klass):
    class PatchedBlock(klass):
        def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):

            # Modifications here
            # r = self._pinfo['r'].pop(0)
            # variant = self._pinfo['variant'].pop(0)
            if self._pinfo['show_shape']: 
                print(x.shape, self._pinfo['class_token'])

            if return_attention:
                return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
            if return_qkv:
                y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
                x = x + self.drop_path(self.gamma_1 * y)
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                return x, qkv

            # Modifications here
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=True)
            qkv = qkv.detach() #.mean(2)
            q, k, v = qkv[0], qkv[1], qkv[2]

            if self.gamma_1 is None:
                x = x + self.drop_path(y)
                x = apply_merge(self._pinfo, x, q, k, v) # Modification
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * y)
                x = apply_merge(self._pinfo, x, q, k, v) # Modification
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x
    return PatchedBlock

def create_transformer_class(klass):
    import torch
    class PatchedNeuralTransformer(klass):

        # Copied from original model defination file
        # We modify this method to collect positional embedding and temporal embedding.
        def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
            batch_size, n, a, t = x.shape
            input_time_window = a if t == self.patch_size else t
            x : torch.Tensor = self.patch_embed(x)

            cls_tokens = self.cls_token #.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

            # x = torch.cat((cls_tokens, x), dim=1)
            # print(x.shape)
            # print((batch_size, n, a, t))
            x = x.reshape((batch_size, n, a, t))
            if self.pos_embed is not None:
                pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
                x = x + pos_embed_used[:, 1:, None, :]
                cls_tokens = cls_tokens + pos_embed_used[:, :1, :]
                # pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
                # pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
                # x = x + pos_embed
            if self.time_embed is not None:
                nc = n if t == self.patch_size else a
                time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1) #.expand(batch_size, nc, -1, -1).flatten(1, 2)
                x = x + time_embed
                # x[:, 1:, :] += time_embed
                # x = torch.cat([x[:, :1, :], x[:, 1:, :] + time_embed], dim=1)

            x = torch.cat([cls_tokens.expand(batch_size, -1, -1), x.flatten(1, 2)], dim=1)

            # embed = pos_embed[:, 1:, :] + time_embed
            # self._pinfo['pte'] = embed.detach()

            x = self.pos_drop(x)
            
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=None)
            
            x = self.norm(x)
            if self.fc_norm is not None:
                if return_all_tokens:
                    return self.fc_norm(x)
                t = x[:, 1:, :]
                if return_patch_tokens:
                    return self.fc_norm(t)
                else:
                    return self.fc_norm(t.mean(1))
            else:
                if return_all_tokens:
                    return x
                elif return_patch_tokens:
                    return x[:, 1:]
                else:
                    return x[:, 0]


        def forward(self, x, *args, **kwargs):
            # handle pinfo related thing here

            depth = len(self.blocks)
            pinfo = self._pinfo

            reset_common_pinfo(self, pinfo, depth)

            # self._pinfo["r"] = parse_r(depth, self.r)
            # self._pinfo["variant"] = parse_variant(depth, self.variant)
            pinfo['shape'] = None
            pinfo["size"] = None
            pinfo["source"] = None
            pinfo["qkv"] = None
            # pinfo["pte"] = None

            # self._pinfo['pivot_factor'] = self.pivot_factor
            self._pinfo["use_cls"] = self.use_cls
            # if hasattr(self, 'pivot_const'):
            #     self._pinfo['pivot_const'] = self.pivot_const

            # bsz, chs, ts, dim = x.shape
            # self._pinfo['raw_raw_data'] = x.reshape(bsz, chs * ts, dim)
            # self._pinfo['raw_data'] = None

            # self._pinfo['raw_dft_data'] = torch.fft.rfft(x.reshape(bsz, chs * ts, dim), dim=-1)[..., :81].abs()
            # self._pinfo['dft_data'] = None

            # self._pinfo["pe_score"] = None
            # self._pinfo["alibi"] = None
            # self._pinfo["attn_score"] = None

            return super().forward(x = x, *args, **kwargs)
    return PatchedNeuralTransformer

def apply_patch(model, trace_source = False, show_shape = False, tome_scheme = False):
    if model.__class__.__name__ != 'NeuralTransformer':
        return
    
    pinfo = {
        'r': 0,
        'variant': '',
        'trace_source': trace_source,
        'source': None,
        'size': None,
        'qkv': None,
        "prop_attn": False,
        "class_token": True,
        "distill_token": False,
        "show_shape": show_shape,
        "tome_scheme": tome_scheme,
    }

    model._pinfo = pinfo
    model.r = 0
    model.variant = ''
    model.pivot_factor = 0.05
    model.use_cls = True

    PatchedNeuralTransformer = create_transformer_class(model.__class__)
    PatchedBlock = create_block_class(model.blocks[0].__class__)

    model.__class__ = PatchedNeuralTransformer
    model._pinfo = pinfo

    for blk in model.blocks:
        blk.__class__ = PatchedBlock
        blk._pinfo = pinfo

    print('Patched LaBraM. Layers={}'.format(len(model.blocks)))

if __name__ == '__main__':
    from timm.models import create_model
    # so that the model is registered in timm, then we can load it via create_model
    from thirdparty.LaBraM import modeling_finetune
    # from thirdparty.LaBraM import utils
    # register_model()
    import torch

    model = create_model(
        'labram_base_patch200_200',
        pretrained=False,
        num_classes=1,
        drop_rate=0,
        drop_path_rate=0.1,
        attn_drop_rate=0,
        drop_block_rate=None,
        use_mean_pooling=False,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )

    ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    ch_names = [it.split(' ')[-1].split('-')[0] for it in ch_names]
    standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

    input_channels = [0]
    for it in ch_names:
        input_channels.append(standard_1020.index(it))
    x = torch.rand(1, 23, 10, 200)
    print(model(x, input_channels))

    apply_patch(model)
    model.r = 1
    model.variant = 'tome'
    print(model(x, input_channels))

