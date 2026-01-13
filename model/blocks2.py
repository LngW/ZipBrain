import torch
from torch import einsum, nn

from .blocks import ToMeBlock, FeedForward, PreNorm, ResBlock

class LinearSelfAttention(nn.Module):
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

class ToMeLinAttn(LinearSelfAttention):
    def __init__(self, dim, d_heads, heads, dropout, *args, **kwargs):
        super().__init__(dim, d_heads, heads, dropout, *args, **kwargs)

    def _generate_output(self, attn, k):
        return attn, k

class LinearEncoderLayer(nn.Module):
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
    
    def forward(self, x, size):
        attn, k = self.attn(x)
        x = x + attn

        x, size = self.block(x, k, size)
        
        x = x + self.ffn(x)
        return x, size

class LinearAttentionEncoder(nn.Module):
    def __init__(self, dim, heads, depth, ff_dropout = 0., attn_dropout = 0., *args, **kwargs):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            ffn = FeedForward(dim, dropout=ff_dropout)
            attn = LinearSelfAttention(dim, heads, dim // heads, attn_dropout)
            layers.append(LinearEncoderLayer(dim, attn, ffn))

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return layer
    
class ToMeEncoder(nn.Module):
    def __init__(self, emb_size, heads, r, num_layers, dropout=0.1, **kwargs):
        super().__init__()
        ffn = FeedForward(emb_size, dropout=0)
        attn = ToMeLinAttn(emb_size, heads, emb_size // heads, dropout)
        self.layers = nn.ModuleList([
            ToMeEncoderLayer(emb_size, attn, ffn, r)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        size = None
        for layer in self.layers:
            x, size = layer(x, size)
        return x
