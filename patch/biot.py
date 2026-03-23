from .utils import parse_r, parse_variant

# def apply_merge(pinfo, r, variant, x):
#     pinfo['qkv'] = None


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


def apply_patch(model, trace_source: bool = False):
    if model.__class__.__name__ != 'BIOTClassifier':
        # we can only apply to BIOTClassifier
        return
    
    biot = model.biot
    transformer = biot.transformer
    sequential = transformer.layers
    self_attn = sequential.layers[0][0].fn

    from .linear_attn_transformer import make_sequential_class, make_self_attention_class
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
    from thirdparty.BIOT.model import BIOTClassifier
    import torch

    x = torch.rand((1, 16, 2000)) # (bsz, chs, ts * freq) => (bsz, 16, 10, 200)

    model = BIOTClassifier(
        n_classes=1,
        # set the n_channels according to the pretrained model if necessary
        n_channels=16,
        n_fft=200,
        hop_length=100,
    )

    print(model(x))

    apply_patch(model)
    model.r = [38]
    model.variant = ['tome']

    print(model(x))