import torch
from torch import nn
from einops import einsum, rearrange
import numpy as np

from .tome_blocks import create_tome_block

def create_biot_encoder_block(
    dim, 
    heads, 
    depth, 
    max_seq_len, 
    attn_layer_dropout, 
    attn_dropout, 
    *,
    ffn_dropout,
    tome_r,
    top_k,
    linear,
    tome_container,
    **kwargs):

    transformer = None

    if linear and top_k == 0 and (len(tome_r) == 0 or all([it == 0 for it in tome_r])):
        from linear_attention_transformer import LinearAttentionTransformer
        transformer = LinearAttentionTransformer(
            dim = dim,
            heads = heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout = attn_layer_dropout,
            attn_dropout = attn_dropout,
            ff_dropout=ffn_dropout,
        )
    else:
        if linear and top_k != 0:
            raise UserWarning("top_k is set with linear, but top_k is not available in linear attention")
        transformer = Encoder(
            depth, 
            tome_r,
            dim = dim, 
            heads = heads, 
            attn_dropout = attn_dropout, 
            linear = linear,
            top_k = top_k,
            ffn_dropout = ffn_dropout,
            tome_container = tome_container,
            **kwargs
        )

    if kwargs.get('cls_token', False):
        class CLSTokenQuery(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = transformer
                with torch.random.fork_rng():
                    self.register_buffer('cls_token', torch.randn(dim, dtype=torch.float, requires_grad=True).view(1, 1, dim))
            
            def forward(self, x):
                emb = self.transformer(x)
                bsz = emb.shape[0]
                emb = torch.nn.functional.scaled_dot_product_attention(
                    self.cls_token.expand(bsz, -1, -1), emb, emb
                )
                return emb
            
        return CLSTokenQuery()
    else:
        return transformer

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, linear = False, top_k = 0, flash = False, *args, **kwargs):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout_p = dropout

        self.n_heads = heads
        self.d_heads = dim // heads
        self.linear = linear
        self.top_k = top_k
        self.flash = not linear and top_k == 0 and flash

    def __sa_attn(self, x, q, k, v):
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.)
        else:
            B, N, D = x.shape
            H, HD = self.n_heads, self.d_heads
            scores = (q * (HD ** -0.5)) @ k.transpose(-1, -2)
            # scores = scores / np.sqrt(HD)

            # if self.top_k > 0 and self.top_k < N:
            #     topk, _ = torch.topk(scores, self.top_k, dim=-1)
            #     scores[scores < topk[..., -1:]] = -torch.inf

            # scores = scores - scores.amax(-1, True)
            weights = torch.softmax(scores, dim = -1)
            if self.dropout_p > 0:
                weights = torch.dropout(weights, self.dropout_p, self.training)
            out = weights @ v

        out = rearrange(out, "b h n d->b n (h d)")
        out = self.out_proj(out)

        return out

    def __li_attn(self, x, q, k, v):
        B, N, D = x.shape
        H, HD = self.n_heads, self.d_heads
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2) # Softmax over Sequence dimension (now dim -2)

        q = q * HD ** -0.5

        context = einsum(k, v, 'b h n d, b h n e->b h d e')
        out = einsum(q, context, 'b h n d, b h d e-> b h n e')
        out = rearrange(out, 'b h n e->b n (h e)')
        out = self.out_proj(out)

        return out

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.n_heads, self.d_heads
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b n (p h d) -> p b h n d", p=3, h=H, d=HD) #.permute(2, 0, 3, 1, 4)

        if self.linear:
            out = self.__li_attn(x, q, k, v)
        else:
            out = self.__sa_attn(x, q, k, v)

        # out = self.out_proj(out)
        # out = self.dropout(out)

        return out, k

class Block(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ffn_dropout, linear, top_k, flash, tome_variant, tome_r, tome_container, *args, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout, linear = linear, top_k = top_k, flash=flash)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(4 * dim, dim)
        )

        self.tome = create_tome_block(tome_variant, tome_container, tome_r, dim)

        print(type(self.tome))
    
    def forward(self, x, size):
        dx, k = self.attn(self.norm1(x))
        x = x + dx

        # if self.tome_r > 0 and self.tome_r < x.shape[1]:
        x, size = self.tome(x, k.detach(), size)

        x = x + self.ffn(self.norm2(x))

        return x, size

class Encoder(nn.Module):
    def __init__(self, depth, tome_r, tome_variant, **kwargs):
        super().__init__()
        layers = nn.ModuleList()

        if len(tome_r) == 0:
            tome_r = [0] * depth
        elif len(tome_r) == 1:
            tome_r = tome_r * depth
        elif len(tome_r) != depth:
            comple = depth - len(tome_r)
            if comple > 0:
                tome_r = tome_r + [0] * comple

        if len(tome_variant) == 0:
            tome_variant = [''] * depth
        elif len(tome_variant) == 1:
            tome_variant = tome_variant * depth
        elif len(tome_variant) != depth:
            comple = depth - len(tome_variant)
            if comple > 0:
                tome_variant = tome_variant + [''] * depth

        for idx in range(depth):
            layers.append(Block(tome_r=tome_r[idx], tome_variant=tome_variant[idx], **kwargs))
        
        self.layers = layers

    def forward(self, x):
        size = None
        for layer in self.layers:
            x, size = layer(x, size)
        return x