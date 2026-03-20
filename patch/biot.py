from patch.utils import parse_r, parse_variant

# def apply_merge(pinfo, r, variant, x):
#     pinfo['qkv'] = None

def apply_merge(pinfo, r : int, variant : str, x):

    q, k, v = pinfo['qkv']
    pinfo['qkv'] = None

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

    from merges import apply_merge_impl
    return apply_merge_impl(pinfo, r, variant, x, metric)

def make_classifier_class(klass):
    class PatchedBIOTClassifier(klass):

        def forward(self, x):
            # generate _pinfo here

            depth = len(self.biot.transformer.layers.layers)
            self._pinfo["r"] = parse_r(depth, self.r)
            self._pinfo["variant"] = parse_variant(depth, self.variant)
            self._pinfo['shape'] = None
            self._pinfo["size"] = None
            self._pinfo["source"] = None
            self._pinfo["qkv"] = None
            self._pinfo["pe_score"] = None
            self._pinfo["alibi"] = None
            self._pinfo["attn_score"] = None

            return super().forward(x)

    return PatchedBIOTClassifier

def make_biot_encoder_class(klass):
    class PatchedBIOTEncoder(klass):
        def stft(self, sample):
            import torch
            spectral = torch.stft( 
                input = sample.squeeze(1),
                n_fft = self.n_fft,
                hop_length = self.hop_length,
                center = False,
                onesided = True,
                return_complex = True,
                window=self.window,
            )
            return torch.abs(spectral)

        def forward(self, x, n_channel_offset=0, perturb=False):
            emb_size = self.patch_embedding.projection.out_features
            bsz, chs, seq = x.shape
            seq = (seq - self.n_fft) / self.hop_length + 1
            self._pinfo['shape'] = [bsz, chs, int(seq), emb_size]

            return super().forward(x, n_channel_offset, perturb)
    
    return PatchedBIOTEncoder

# def make_transformer(klass):
#     class PatchedLinearAttentionTransfromer(klass):
        # def __init__(self, **kwargs):
        #     super().__init__(**kwargs)
        #     self.ff_dropout = kwargs.get('ff_dropout', 0)

        #     self.layers.

def make_sequential_class(klass):
    class PatchedSequentialSequence(klass):
        def forward(self, x, **kwargs):
            from linear_attention_transformer.reversible import route_args, layer_drop
            args = route_args(self.args_route, kwargs, len(self.layers))

            # modification start
            layers_and_args = list(zip(self.layers, args, self._pinfo['r'], self._pinfo['variant']))
            # r = self._pinfo['r'].pop(0)
            # variant = self._pinfo['variant'].pop(0)
            # modification end

            if self.training and self.layer_dropout > 0:
                layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

            # modified this line: add r and variant
            for (f, g), (f_args, g_args), r, variant in layers_and_args:
                x = x + f(x, **f_args)
                # insertation here
                x = apply_merge(self._pinfo, r, variant, x)
                # insertation end
                x = x + g(x, **g_args)
            return x
        
    return PatchedSequentialSequence

def make_self_attention_class(klass):

    class PatchedSelfAttention(klass):
        def forward(self, x, input_mask = None, context = None, context_mask = None, pos_emb = None, **kwargs):
            from linear_attention_transformer.linear_attention_transformer import exists, apply_rotory_pos_emb, split_at_index
            import torch
            from functools import partial

            assert not (self.receives_context and not exists(context)), 'context must be supplied if self attention is in receives context mode'

            if not self.receives_context:
                q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
            else:
                q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

            b, t, e, h, dh = *q.shape, self.heads, self.d_heads

            merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

            q, k, v = map(merge_heads, (q, k, v))

            if exists(pos_emb) and not self.receives_context:
                q, k = apply_rotory_pos_emb(q, k, pos_emb)

            out = []

            split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

            (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

            has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

            if has_local:
                local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
                out.append(local_out)

            if has_global:
                kv_mask = input_mask if not self.receives_context else context_mask
                global_out = self.global_attn_fn(q, k, v, kv_mask = kv_mask)
                out.append(global_out)

            attn = torch.cat(out, dim=1)
            attn = attn.transpose(1, 2).reshape(b, t, -1)

            # modification
            self._pinfo['qkv'] = [q.mean(1), k.mean(1), v.mean(1)]
            # modification End

            return self.dropout(self.to_out(attn))
    
    return PatchedSelfAttention

def apply_patch(model, trace_source: bool = False):
    if model.__class__.__name__ != 'BIOTClassifier':
        # we can only apply to BIOTClassifier
        return
    
    biot = model.biot
    transformer = biot.transformer
    sequential = transformer.layers
    self_attn = sequential.layers[0][0].fn

    PatchedClassifier = make_classifier_class(model.__class__)
    PatchedEncoder = make_biot_encoder_class(biot.__class__)
    PatchedSequential = make_sequential_class(sequential.__class__)
    PatchedAttention = make_self_attention_class(self_attn.__class__)

    import torch

    model.__class__ = PatchedClassifier
    model.r = 0
    model.variant = None
    _pinfo = model._pinfo = {
        "r": model.r,
        "variant": model.variant,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": False,
        "class_token": False,
        "distill_token": False,
        "pe" : model.biot.positional_encoding.pe,
        "pe_score": None,
    }

    biot.__class__ = PatchedEncoder
    biot.register_buffer('window', torch.ones(biot.n_fft))
    biot._pinfo = _pinfo

    sequential.__class__ = PatchedSequential
    sequential._pinfo = _pinfo

    for sub in sequential.layers:
        if sub[0].fn.__class__.__name__ == 'SelfAttention':
            sub[0].fn.__class__ = PatchedAttention
            sub[0].fn._pinfo = _pinfo
    
    print("Patched BIOTClassifier")


if __name__ == '__main__':
    from model import BIOTClassifier
    import torch

    x = torch.rand((1, 16, 2000))

    model = BIOTClassifier(
        n_classes=1,
        # set the n_channels according to the pretrained model if necessary
        n_channels=16,
        n_fft=200,
        hop_length=100,
    )

    apply_patch(model)
    model.r = [38]
    model.variant = ['tome']

    print(model(x))