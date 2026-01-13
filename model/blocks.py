import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads, dropout = 0.1, *args, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)

        self.proj_o = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)

    def _score_masking(self, score, q, k, v):
        return score
    
    def _generate_returns(self, out, score, q, k, v):
        return out

    def forward(self, x):
        # x: (batch, seq_len, emb_size)
        batch_size, seq_len, _ = x.shape

        q = self.query(x).view(batch_size, seq_len, self.heads, self.emb_size // self.heads)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.emb_size // self.heads)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.emb_size // self.heads)

        # (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        # (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.emb_size // self.heads)**0.5

        scores = self._score_masking(scores, q, k, v)
        
        attention_weights = self.softmax(scores)
        out = torch.matmul(attention_weights, v)

        # Concatenate heads and put through final linear layer (if needed)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_size)
        out = self.proj_o(out)
        out = self.drop(out)

        return self._generate_returns(out, scores, q, k, v)

class TopKSelfAttention(MultiHeadAttention):
    def __init__(self, emb_size, heads, k, dropout, *args, **kwargs):
        super().__init__(emb_size=emb_size, heads=heads, dropout=dropout, *args, **kwargs)
        self.k = k

    def _score_masking(self, scores, q, k, v):
        # Apply Top-K masking
        # For each query, select the top-k keys
        # If k <= 0, behaves as a vanilla multihead self-attention
        if self.k > 0:
            topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)

            # Create a mask for the top-k elements
            # Initialize with a very small negative number to be ignored by softmax
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(-1, topk_indices, 0) # Set top-k positions to 0

            # Apply mask to scores
            scores = scores + mask
        return scores

class ToMeAttention(MultiHeadAttention):
    def __init__(self, emb_size, heads, dropout=0.1, *args, **kwargs):
        super().__init__(emb_size, heads, dropout, *args, **kwargs)

    def _generate_returns(self, out, score, q, k, v):
        return out, k

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
    
class FeedForward(nn.Module):
    def __init__(self, emb_size, ffn_hidden_size, ffn_dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w1 = nn.Linear(emb_size, ffn_hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(ffn_dropout)
        self.w2 = nn.Linear(ffn_hidden_size, emb_size)

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
        return self.fn(self.norm(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return x + self.fn(x)

class TopKEncoderLayer(nn.Module):
    def __init__(self, emb_size, heads, k, ffn_hidden_size, dropout=0.1):
        super().__init__()
        attn = TopKSelfAttention(emb_size, heads, k, dropout)
        ffn = FeedForward(emb_size, ffn_hidden_size, 0)

        self.attention = ResidualBlock(PreNorm(emb_size, attn))
        self.ffn = ResidualBlock(PreNorm(emb_size, ffn))

    def forward(self, x):
        # Self-attention part
        x = self.attention(x)

        # FFN part
        x = self.ffn(x)
        return x

class ToMeEncoderLayer(nn.Module):
    def __init__(self, emb_size, heads, r, ffn_hidden_size, dropout=0.1):
        super().__init__()
        attn = ToMeAttention(emb_size, heads, dropout)
        ffn = FeedForward(emb_size, ffn_hidden_size, 0)

        self.attention = PreNorm(emb_size, attn)
        self.block = ToMeBlock(r)
        self.ffn = PreNorm(emb_size, ffn)

    def forward(self, x, size):
        # Self-attention part
        x_, k = self.attention(x)
        x = x + x_

        # ToMe part
        x, size = self.block(x, k, size)

        # FFN part
        ffn_output = self.ffn(x)
        x = x + ffn_output

        return x, size

class TopKEncoder(nn.Module):
    def __init__(self, emb_size, heads, k, ffn_hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()

        # self.layers.append(nn.LayerNorm(emb_size))
        self.layers.extend([
            TopKEncoderLayer(emb_size, heads, k, ffn_hidden_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ToMeEncoder(nn.Module):
    def __init__(self, emb_size, heads, r, ffn_hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ToMeEncoderLayer(emb_size, heads, r, ffn_hidden_size, dropout)
            for _ in range(num_layers)
        ])

        # self._tome_info = {
        #     'size': None
        # }

        # for layer in self.layers:
        #     layer.block._tome_info = self._tome_info

    def forward(self, x):
        # self._tome_info['size'] = None
        size = None
        for layer in self.layers:
            x, size = layer(x, size)
        return x
