import torch
from torch import einsum, nn
# from linear_attention_transformer.linear_attention_transformer import Chunk
# from local_attention import LocalAttention

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
        q = self.to_q(x).view(bsz, seq, self.heads, -1).transpose(1, 2)
        k = self.to_k(x).view(bsz, seq, self.heads, -1).transpose(1, 2)
        v = self.to_v(x).view(bsz, seq, self.heads, -1).transpose(1, 2)

        dim = q.shape[-1]

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2) # Softmax over Sequence dimension (now dim -2)

        q = q * dim ** -0.5

        context = einsum('bhnd,bhne->bhde', k, v)
        attn = einsum('bhnd,bhde->bhne', q, context)
        # context = torch.matmul(k.transpose(-1, -2), v)
        # attn = torch.matmul(q, context)
        attn = attn.transpose(1, 2).reshape(bsz, seq, -1)
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

class LinearAttentionEncoder(nn.Module):
    def __init__(self, dim, heads, depth, ff_dropout = 0., attn_dropout = 0., *args, **kwargs):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            ffn = FeedForward(dim, ff_dropout)
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