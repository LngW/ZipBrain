import torch
from torch import nn
from einops import einsum, rearrange
import numpy as np

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
    flash=False,
    rtl_tome=False,
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
        return Encoder(dim, heads, depth, attn_layer_dropout, linear, top_k, tome_r, flash, rtl_tome)

def ToMeBlock(r = 0, *args, **kwargs):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum

    class ToMeBlock(nn.Module):
        def __init__(self, r = 2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r = r

        def forward(self, x, k, size_old):
            bsz, hd, seq, _ = k.shape
            _, merge, _ = bipartite_soft_matching(
                k.transpose(1,2).reshape(bsz, seq, -1),
                self.r
            )

            x, size = merge_wavg_sum(merge, x, size_old)

            return x, size
        
    return ToMeBlock(r = r, *args, **kwargs)

def RunTimeLengthToMeBlock(r = 0, emb_dim = 256, *args, **kwargs):
    # from .tome.merge import bipartite_soft_matching, merge_wavg_sum

    class ToMeBlock(nn.Module):
        def __init__(self, r, emb_dim, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r = r
            self.consec = nn.Embedding(1024, emb_dim)
            # self.skip = nn.Embedding(1024, emb_dim)

        # we need the block go back to shape of (B, C, N, D)
        def forward(self, x : torch.Tensor, k, size_old):
            x = x.detach()
            bsz, c_s, dim = x.shape
            channels = 16
            seq = c_s // 16
            k = k.detach().transpose(1,2).reshape(bsz, channels, seq, dim)

            k = k.reshape(bsz * channels, seq, dim)
            # normed = torch.linalg.norm(k, dim=-1, keepdim=True)
            k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
            scores : torch.Tensor = (k[:, :-1] * k[:, 1:]).sum(-1)

            kval, kidx = torch.topk(scores, r, sorted=False)
            kidx, _ = kidx.sort(-1)

            cvt = torch.arange(seq, device=k.device)[None].repeat(bsz * channels, 1)
            for b in range(bsz * channels):
                for i in range(r):
                    t = kidx[b, i]
                    k = cvt[b, t + 1] = cvt[b, t]

            eye = torch.eye(seq, dtype=torch.long, device=k.device)[None].repeat(bsz * channels, 1, 1)
            proj = torch.gather(eye, -1, cvt[..., None, :].expand(-1, seq, -1))
            sums = proj.sum(-1, True)
            proj = proj / torch.maximum(sums, torch.ones_like(sums))

            nidx = torch.empty(bsz * channels, seq - r, dtype=torch.long, device=k.device)
            for b in range(nidx.shape[0]):
                j = 0
                for i in range(seq):
                    if sums[b, i, 0] > 0:
                        nidx[b, j] = i
                        j = j + 1

            rtl = torch.gather(sums.squeeze(-1), -1, nidx)
            proj = torch.gather(proj, -2, nidx[..., None].expand(-1, -1, seq))

            y0 = self.consec(rtl).view(bsz, channels, seq-r, dim)
            y1 = proj.view(bsz, channels, seq-r, seq) @ x.view(bsz, channels, seq, dim)

            y = y0 + y1
            return y.view(bsz, channels * seq, dim), None

    return ToMeBlock(r, emb_dim)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, linear = False, top_k = 0, flash = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            scores = q @ k.transpose(-1, -2)
            scores = scores / np.sqrt(HD)

            if self.top_k > 0 and self.top_k < N:
                topk, _ = torch.topk(scores, self.top_k, dim=-1)
                scores[scores < topk[..., -1:]] = -torch.inf

            scores = scores - scores.amax(-1, True)
            weights = torch.softmax(scores, dim = -1)
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
    def __init__(self, dim, heads, dropout, linear = False, top_k = 0, tome_r = 0, flash = False, rtl_tome=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, linear = linear, top_k = top_k, flash=flash)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        if rtl_tome:
            self.tome = ToMeBlock(tome_r)
        else:
            self.tome = RunTimeLengthToMeBlock(tome_r, dim)
    
    def forward(self, x, size):
        dx, k = self.attn(self.norm1(x))
        x = x + dx

        # if self.tome_r > 0 and self.tome_r < x.shape[1]:
        x, size = self.tome(x, k.detach(), size)

        x = x + self.ffn(self.norm2(x))

        return x, size

class Encoder(nn.Module):
    def __init__(self, dim, heads, depth, dropout, linear = False, top_k = 0, tome_r = 0, flash=False, rtl_tome=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = nn.ModuleList()

        for _ in range(depth):
            layers.append(Block(dim, heads, dropout, linear = linear, top_k=top_k, tome_r=tome_r, flash=flash, rtl_tome=False))
        
        self.layers = layers

    def forward(self, x):
        size = None
        for layer in self.layers:
            x, size = layer(x, size)
        return x