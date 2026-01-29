import torch
from torch import nn

def create_biot_encoder_block(
    dim, 
    heads, 
    depth, 
    max_seq_len, 
    attn_layer_dropout, 
    attn_dropout, 
    *,
    tome_r=0,
    top_k=0,
    linear=False,
    **kwargs):

    if linear and tome_r == 0 and top_k == 0:
        from linear_attention_transformer import LinearAttentionTransformer
        return LinearAttentionTransformer(
            dim = dim,
            heads = heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout = attn_layer_dropout,
            attn_dropout = attn_dropout,
        )
    else:
        if linear and top_k != 0:
            raise UserWarning("top_k is set with linear, but top_k is not available in linear attention")
        return Encoder(dim, heads, depth, attn_layer_dropout, linear, top_k, tome_r)

def ToMeBlock(r = 2, *args, **kwargs):
    from .tome.merge import bipartite_soft_matching, merge_wavg

    class ToMeBlock(nn.Module):
        def __init__(self, r = 2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r = r

        def forward(self, x, k, size_old):
            bsz, hd, seq, _ = k.shape
            merge, _ = bipartite_soft_matching(
                k.transpose(1,2).reshape(bsz, seq, -1),
                self.r
            )

            x, size = merge_wavg(merge, x, size_old)

            return x, size
        
    return ToMeBlock(r = r, *args, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, linear = False, top_k = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.n_heads = heads
        self.d_heads = dim // heads
        self.linear = linear
        self.top_k = top_k

    def __sa_attn(self, x, q, k, v):
        B, N, D = x.shape
        H, HD = self.n_heads, self.d_heads
        scores = q @ k.transpose(-1, -2)

        if self.top_k > 0 and self.top_k < N:
            top_k_values, _ = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            scores[scores < top_k_values.min(dim=-1, keepdim=True)] = -torch.inf

        weights = torch.softmax(scores, dim = -1)
        out = weights @ v
        out = out.transpose(1, 2).flatten(-2)

        return out

    def __li_attn(self, x, q, k, v):
        from torch import einsum
        B, N, D = x.shape
        H, HD = self.n_heads, self.d_heads
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2) # Softmax over Sequence dimension (now dim -2)

        q = q * HD ** -0.5

        context = einsum('bhnd,bhne->bhde', k, v)
        out = einsum('bhnd,bhde->bhne', q, context)
        out = einsum('b h n e -> b n (h e)')

        return out

    def forward(self, x):
        B, N, D = x.shape
        H, HD = self.n_heads, self.d_heads
        qkv = self.qkv_proj(x).view(B, N, 3, H, HD)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.linear:
            out = self.__li_attn(x, q, k, v)
        else:
            out = self.__sa_attn(x, q, k, v)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out, k

class Block(nn.Module):
    def __init__(self, dim, heads, dropout, linear = False, top_k = 0, tome_r = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, linear = linear, top_k = top_k)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        self.tome = ToMeBlock(tome_r)
    
    def forward(self, x, size):
        dx, k = self.attn(self.norm1(x))
        x = x + dx

        # if self.tome_r > 0 and self.tome_r < x.shape[1]:
        x, size = self.tome(x, k, size)

        x = x + self.ffn(self.norm2(x))

        return x, size

class Encoder(nn.Module):
    def __init__(self, dim, heads, depth, dropout, linear = False, top_k = 0, tome_r = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = nn.ModuleList()

        for _ in range(depth):
            layers.append(Block(dim, heads, dropout, top_k, linear))
        
        self.layers = layers

    def forward(self, x):
        size = None
        for layer in self.layers:
            x, size = layer(x, size)
        return x