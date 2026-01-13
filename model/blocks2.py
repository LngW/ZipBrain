import torch
from torch import einsum, nn

from .blocks import ToMeBlock

class MHSelfAttention(nn.Module):
    def __init__(self, dim, d_heads, heads, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads

        # self.local_attn = LocalAttention(128, False, 0.2)

        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_k = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, seq = x.shape[:2]
        # Transpose to (Batch, Heads, Seq, Dim) to mix over Sequence
        q = self.to_q(x).reshape(bsz, seq, self.heads, -1).transpose(1, 2)
        k = self.to_k(x).reshape(bsz, seq, self.heads, -1).transpose(1, 2)
        v = self.to_v(x).reshape(bsz, seq, self.heads, -1).transpose(1, 2)

        dim = q.shape[-1]

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2) # Softmax over Sequence dimension (now dim -2)

        q = q * dim ** -0.5

        context = einsum('bhnd,bhne->bhde', k, v)
        attn = einsum('bhnd,bhde->bhne', q, context)
        attn = attn.transpose(1, 2).reshape(bsz, seq, -1)
        attn = self.to_out(attn)
        attn = self.dropout(attn)

        return self._generate_output(attn, k)
    
    def _generate_output(self, attn, k):
        return attn

class ToMeLinAttn(MHSelfAttention):
    def __init__(self, dim, d_heads, heads, dropout, *args, **kwargs):
        super().__init__(dim, d_heads, heads, dropout, *args, **kwargs)

    def _generate_output(self, attn, k):
        return attn, k

class FeedForward(nn.Module):
    def __init__(self, dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, attn, ffn, *args, **kwargs):
        super().__init__()
        self.attn = PreNorm(dim, attn)
        self.ffn = PreNorm(dim, ffn)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class ToMeEncoderLayer(nn.Module):
    def __init__(self, dim, attn, ffn, r = 2, *args, **kwargs):
        super().__init__()
        self.attn = PreNorm(dim, attn)
        self.block = ToMeBlock(r)
        self.ffn = PreNorm(dim, ffn)
    
    def forward(self, x):
        attn, k = self.attn(x)
        x = x + attn

        x = self.block(x, k)
        
        x = x + self.ffn(x)
        return x

class LinearAttentionEncoder(nn.Module):
    def __init__(self, dim, heads, depth, ff_dropout = 0., attn_dropout = 0., *args, **kwargs):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            ffn = FeedForward(dim, dropout=ff_dropout)
            attn = MHSelfAttention(dim, heads, dim // heads, attn_dropout)
            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ffn)
            ]))

        self.layers = layers

    def forward(self, x):
        for (f, g) in self.layers:
            x = x + f(x)
            x = x + g(x)
        return x
    
class ToMeEncoder(nn.Module):
    def __init__(self, emb_size, heads, r, num_layers, dropout=0.1, **kwargs):
        super().__init__()
        ffn = FeedForward(emb_size, dropout=0)
        attn = ToMeLinAttn(emb_size, heads, emb_size // heads, dropout)
        self.layers = nn.ModuleList([
            ToMeEncoderLayer(emb_size, attn, ffn, r)
            for _ in range(num_layers)
        ])

        self._tome_info = {
            'size': None
        }

        for layer in self.layers:
            layer.block._tome_info = self._tome_info

    def forward(self, x):
        self._tome_info['size'] = None
        for layer in self.layers:
            x = layer(x)
        return x
