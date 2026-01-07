import torch
from torch import nn

class MHSelfAttention(nn.Module):
    def __init__(self, dim, d_heads, heads, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads

        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_k = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * heads, bias = False)
        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, seq = x.shape[:2]
        q = self.to_q(x).view(bsz, seq, self.heads, -1)
        k = self.to_k(x).view(bsz, seq, self.heads, -1)
        v = self.to_v(x).view(bsz, seq, self.heads, -1)

        dim = q.shape[-1]

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * dim ** -0.5

        context = torch.matmul(k.transpose(-1, -2), v)

        attn = torch.matmul(q, context)
        attn = attn.view(bsz, seq, -1)
        attn = self.to_out(attn)
        attn = self.dropout(attn)

        return attn

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

# class EncoderLayer(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, attn, ffn, *args, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = attn
        self.ffn = ffn
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LinearAttentionEncoder(nn.Module):
    def __init__(self, dim, heads, depth, ff_dropout = 0., attn_dropout = 0., *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim,
                MHSelfAttention(dim, heads, dim // heads, attn_dropout),
                FeedForward(dim, ff_dropout),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x