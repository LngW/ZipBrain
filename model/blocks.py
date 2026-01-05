import torch
from torch import nn

class TopKSelfAttention(nn.Module):
    def __init__(self, emb_size, heads, k):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.k = k

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)

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
        
        attention_weights = self.softmax(scores)
        out = torch.matmul(attention_weights, v)

        # Concatenate heads and put through final linear layer (if needed)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_size)
        return out

class TopKEncoderLayer(nn.Module):
    def __init__(self, emb_size, heads, k, ffn_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = TopKSelfAttention(emb_size, heads, k)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, emb_size),
            nn.Dropout(dropout)
        )
        self.attn_layer_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention part
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        # FFN part
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

class TopKEncoder(nn.Module):
    def __init__(self, emb_size, heads, k, ffn_hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TopKEncoderLayer(emb_size, heads, k, ffn_hidden_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VanillaEncoder(nn.Module):
    def __init__(self, emb_size, heads, ffn_hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TopKEncoderLayer(emb_size, heads, 0, ffn_hidden_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
