from .utils import parse_r, parse_variant
import torch

def apply_merge(pinfo, x):
    q, k, v = pinfo['qkv']
    pinfo['qkv'] = None

    r = pinfo['r'].pop(0)
    variant = pinfo['variant'].pop(0)

    if r <= 0:
        return x

    fast_dict = {'q': q, 'k': k, 'v': v, 'x': x}
    def _find_metric(variant):
        left = variant.find('[')
        right = variant.find(']')
        if left > 0 and right > left:
            return fast_dict.get(variant[left + 1 : right], k)
        else:
            return k
    
    if variant is not None:
        metric = _find_metric(variant)

    embed_num = pinfo['embed_num']
    x_raw, x_summary = x[:, :-embed_num], x[:, -embed_num:]
    metric_raw, metric_summary = metric[:, :-embed_num], metric[:, -embed_num:]

    from merges import apply_merge_impl
    x_raw = apply_merge_impl(pinfo, r, variant, x_raw, metric_raw)

    import torch
    return torch.cat([x_raw, x_summary], dim=-2)


def make_attention_class(klass):
    import torch
    import math

    from thirdparty.EEGPT.downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import apply_rotary_emb
    class PatchedAttention(klass):
        def forward(self, x, freqs=None):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3,B,nh,t,d
            q, k, v = qkv[0], qkv[1], qkv[2] # B,nh,t,d
            if self.use_rope:# RoPE
                q = apply_rotary_emb(freqs, q)
                k = apply_rotary_emb(freqs, k)
            if self.return_attention:
                if self.is_causal:
                    attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                    attn_maak = torch.zeros(q.size(-2), q.size(-2))
                    attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                    attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
                else:
                    attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
                return attn_weight
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
            x = y.transpose(1, 2).contiguous().view(B, T, C) #(B, nh, T, hs) -> (B, T, hs*nh)
            x = self.proj(x)
            x = self.proj_drop(x)
            self._pinfo['qkv'] = q.detach().mean(1), k.detach().mean(1), v.detach().mean(1) # One Line Modification
            return x
    return PatchedAttention

def make_block_class(klass):
    class PatchedBlock(klass):
        def forward(self, x, freqs=None):
            y = self.attn(self.norm1(x), freqs)
            if self.return_attention: return y
            x = x + self.drop_path(y)
            # Modification here
            x = apply_merge(self._pinfo, x)
            # Modification end
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    return PatchedBlock

def make_classifier_class(klass):
    class PatchedClassifier(klass):
        def forward(self, *args, **kwargs):
            depth = len(self.target_encoder.blocks)
            self._pinfo["r"] = parse_r(depth, self.r)
            self._pinfo["variant"] = parse_variant(depth, self.variant)
            self._pinfo['shape'] = None
            self._pinfo["size"] = None
            self._pinfo["source"] = None
            self._pinfo["qkv"] = None
            # self._pinfo["pe_score"] = None
            # self._pinfo["alibi"] = None
            # self._pinfo["attn_score"] = None

            return super().forward(*args, **kwargs)

    return PatchedClassifier



def apply_patch(model, trace_source = False):
    if model.__class__.__name__ != 'EEGPTClassifier':
        raise NotImplementedError("Unrecognized model class: {}".format(model.__class__.__name__))
    
    PatchedClassifier = make_classifier_class(model.__class__)
    PatchedBlock = make_block_class(model.target_encoder.blocks[0].__class__)
    PatchedAttention = make_attention_class(model.target_encoder.blocks[0].attn.__class__)

    pinfo = {
        'r': 0,
        'variant': '',
        'class_token': False,
        'distill_token': False,
        "prop_attn": False,
        'size': None,
        'trace_source': trace_source,
        'source': None,
        'embed_num': model.target_encoder.embed_num,
    }

    model.__class__ = PatchedClassifier
    model.r = 0
    model.variant = ''
    model._pinfo = pinfo

    for blk in model.target_encoder.blocks:
        blk.__class__ = PatchedBlock
        blk._pinfo = pinfo

        blk.attn.__class__ = PatchedAttention
        blk.attn._pinfo = pinfo

    print('Patched EEGPT, Layers = {}'.format(len(model.target_encoder.blocks)))

if __name__ == '__main__':
    from thirdparty.EEGPT.downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
    use_channels_names = [      
             'FP1','FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]

    model = EEGPTClassifier(
        num_classes=1,
        in_channels=len(use_channels_names), 
        img_size=[len(use_channels_names),2000], 
        use_channels_names=use_channels_names, 
        use_chan_conv=True,
        )

    import torch

    x = torch.rand(1, 20, 2000)
    model.eval()
    print(model(x))

    apply_patch(model)

    model.r = 1
    model.variant = 'tome[k]'

    print(model(x))