def apply_merge(pinfo, x):
    q, k, v = pinfo['qkv']
    del pinfo['qkv']

    r = pinfo['r'].pop(0)
    variant = pinfo['variant'].pop(0)

    from merges import apply_merge_impl

    return apply_merge_impl(pinfo, r, variant, x, q, k, v)


def make_patch_vit(klass):
    from .utils import reset_common_pinfo
    class PatchedVisionTransformer(klass):
        def forward(self, *args, **kwargs):

            # generate pinfo
            depth = len(self.blocks)

            reset_common_pinfo(self, self._pinfo, depth)

            # self._pinfo["r"] = parse_r(depth, self.r)
            # self._pinfo["variant"] = parse_variant(depth, self.variant)
            self._pinfo['shape'] = None
            self._pinfo["size"] = None
            self._pinfo["source"] = None
            self._pinfo["qkv"] = None
            self._pinfo["pe_score"] = None
            self._pinfo["alibi"] = None
            self._pinfo["attn_score"] = None

            return super().forward(*args, **kwargs)

    return PatchedVisionTransformer

def make_patch_block(klass):
    class PatchedBlock(klass):
        def forward(self, x):
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = apply_merge(self._pinfo, x)
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
    return PatchedBlock

def make_patch_attention(klass):
    import torch
    import torch.nn.functional as F
    class PatchedAttention(klass):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            self._pinfo['qkv'] = q,k,v

            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    return PatchedAttention

def apply_patch(model, trace_source: bool = False, show_shape = False, tome_scheme = False):
    if model.__class__.__name__ != 'VisionTransformer':
        return

    PatchedVisionTransformer = make_patch_vit(model.__class__)
    PatchedBlock = make_patch_block(model.blocks[0].__class__)
    PatchedAttention = make_patch_attention(model.blocks[0].attn.__class__)

    pinfo = {
        'cls_token': True,
        'distill_token': False,
        'r': 0,
        'variant': '',
        'trace_source': trace_source,
        'show_shape': show_shape,
        'tome_scheme': tome_scheme,
    }
    model.r = 0
    model.variant = ''
    model._pinfo = pinfo

    model.__class__ = PatchedVisionTransformer
    for blk in model.blocks:
        if blk.__class__.__name__ == 'Block':
            blk.__class__ = PatchedBlock
            blk._pinfo = pinfo

            blk.attn.__class__ = PatchedAttention
            blk.attn._pinfo = pinfo

    print('Patched STEEGFormer, Layers = {}'.format(len(model.blocks)))