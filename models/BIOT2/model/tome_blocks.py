from functools import partial
import math
import torch
from torch import nn
from einops import rearrange

def create_tome_block(variant : str, container, r, dim):
    if variant == 'random':
        return RandomBlock(container, r, dim)
    if variant == 'tome':
        return ToMeBlock(container, r, dim, use_x=False)
    elif variant == 'tomex':
        return ToMeBlock(container, r, dim, use_x=True)
    elif variant == 'channel':
        return ChannelToMeBlock(container, r, dim)
    elif variant == 'time':
        return TimestepToMeBlock(container, r, dim)
    elif variant == 'f_tome':
        return FullToMeBlock(container, r, dim, use_x=False)
    elif variant == 'fx_tome':
        return FullToMeBlock(container, r, dim, use_x=True)
    elif variant == 'fch_tome':
        return FullChannelToMeBlock(container, r, dim, use_x=False)
    elif variant == 'fxch_tome':
        return FullChannelToMeBlock(container, r, dim, use_x=True)
    elif variant == 'fts_tome':
        return FullTimestepToMeBlock(container, r, dim, use_x=False)
    elif variant == 'fxts_tome':
        return FullTimestepToMeBlock(container, r, dim, use_x=True)
    elif variant.startswith('l_') or variant.startswith('lq_'):
        q_only = variant.startswith('lq_')
        configs = variant.split('_')
        gct = 'global'
        merge = 0
        loss = 0
        attached = True
        if len(configs) > 1:
            gct = configs[1]
        if len(configs) > 2:
            merge = configs[2]
        if len(configs) > 3:
            loss = int(configs[3])
        if len(configs) > 4:
            attached = configs[4] != "d"

        if gct == 'tome' or gct == 'global':
            return LearnableToMeBlock(container, r, dim, q_only, merge, loss, attached)
        elif variant == 'channel':
            return LearnableChannel(container, r, dim, q_only)
        elif variant == 'time':
            return LearnableTimestep(container, r, dim, q_only)
    elif variant == 'rtl':
        return RunTimeLengthToMeBlock(container, r, dim)
    elif variant == 'w_channel':
        return DoubleChannelToMeBlock(container, r, dim)
    elif variant == 'w_time':
        return DoubleTimestepToMeBlock(container, r, dim)
    elif variant == 'l_ctime':
        return LearnableConsectiveTimestepToMeBlock(container, r, dim)
    elif variant == 'layernorm':
        return NormOnly(dim)
    

    return lambda x, k, size: (x, size)

def softsort(x : torch.Tensor, metric : torch.Tensor, tau : float, descending : bool):
    val, idx = torch.sort(metric, dim=-2, descending=False)
    dist = -torch.square(metric.transpose(-2, -1) - val)
    weight = torch.softmax(dist / tau, dim = -1)
    return weight @ x

# @torch.jit.script
def softtopk(x : torch.Tensor, metric : torch.Tensor, k : int, tau : float, largest : bool = True, sorted : bool = True):
    val, idx = torch.topk(metric, k, dim=-1, largest=largest, sorted=sorted)
    weight = torch.softmax(-torch.square(metric[..., None, :] - val[..., None]) / tau, dim = -1)
    return torch.bmm(weight, x), val, idx

def NormOnly(dim):
    class NormOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
        
        def forward(self, x, k, size):
            return self.norm(x), size
    
    return NormOnly()

def RandomBlock(tome_container, r, dim):
    class RandomBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r
            self.dim = dim
        
        def forward(self, x, k, size_old):
            bsz, seq, dim = x.shape
            with torch.random.fork_rng(x.device):
                tmp = torch.randperm(seq)
                src_idx = tmp[:r].view(1, -1, 1).expand(bsz, -1, dim)
                tar_idx = tmp[r:].view(1, -1, 1).expand(bsz, -1, dim)
            
            src = torch.gather(x, -2, src_idx)
            tar = torch.gather(x, -2, tar_idx)

            q = tar / tar.norm(dim=-1, keepdim=True)
            k = src / src.norm(dim=-1, keepdim=True)

            

def ToMeBlock(tome_container, r, dim, use_x=False):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum

    class ToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r
            self.use_x = use_x

        def forward(self, x, k, size_old):
            # bsz, hd, seq, _ = k.shape
            if self.use_x:
                k_ = x.detach()
            else:
                k_ = rearrange(k.detach(), "b h s d -> b s (h d)")

            _, merge, _ = bipartite_soft_matching(
                # k.transpose(1,2).reshape(bsz, seq, -1),
                k_,
                self.r
            )

            x, size = merge_wavg_sum(merge, x, size_old)

            return x, size
        
    return ToMeBlock()

def FullToMeBlock(tome_container, r, dim, use_x=False):
    class ToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r
            self.use_x = use_x

        def forward(self, x, k, size_old):
            if size_old is None:
                size_old = torch.ones_like(x[..., 0:1])
            
            if self.use_x:
                k_ = x.detach()
            else:
                k_ = rearrange(k.detach(), "b h s d -> b s (h d)")
            return _qk_merge(self.r, x, k_, k_, size_old)
        
    return ToMeBlock()

def FullChannelToMeBlock(tome_container, r, dim, use_x=False):
    class ToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r
            self.tome_container = tome_container
            self.use_x = use_x
        
        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']
            wrap = partial(rearrange, pattern="b (c s) d -> (b s) c d", s=seq)
            unwrap = partial(rearrange, pattern="(b s) c d -> b (c s) d", s=seq)

            x = wrap(x)
            size_old = wrap(size_old) if size_old is not None else torch.ones_like(x[..., 0:1])

            if self.use_x:
                k_ = x.detach()
            else:
                k_ = rearrange(k.detach(), "b h (c s) d -> (b s) c (h d)", c=chs, s=seq)

            x_, size_ = _qk_merge(self.r, x, k_, k_, size_old)

            chs = x_.shape[1]
            self.tome_container['shape'] = bsz, chs, seq, dim
            # x_ = rearrange(x_, "(b s) c d -> b (c s) d", c=chs, s=seq)
            # size_ = rearrange(size_, "(b s) c d -> b (c s) d", c=chs, s=seq)

            return unwrap(x_), unwrap(size_)

    return ToMeBlock()

def FullTimestepToMeBlock(tome_container, r, dim, use_x=False):
    class ToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r
            self.tome_container = tome_container
            self.use_x = use_x
        
        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']
            wrap = partial(rearrange, pattern="b (c s) d -> (b c) s d", c=chs)
            unwrap = partial(rearrange, pattern="(b c) s d -> b (c s) d", c=chs)

            x = wrap(x)
            size_old = wrap(size_old) if size_old is not None else torch.ones_like(x[..., 0:1])

            if self.use_x:
                k_ = x.detach()
            else:
                k_ = rearrange(k.detach(), "b h (c s) d -> (b c) s (h d)", c=chs, s=seq)

            x_, size_ = _qk_merge(self.r, x, k_, k_, size_old)

            seq = x_.shape[1]
            self.tome_container['shape'] = bsz, chs, seq, dim
            # x_ = rearrange(x_, "(b s) c d -> b (c s) d", c=chs, s=seq)
            # size_ = rearrange(size_, "(b s) c d -> b (c s) d", c=chs, s=seq)

            return unwrap(x_), unwrap(size_)

    return ToMeBlock()

def ChannelToMeBlock(tome_container, r, dim):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum
    class ChannelToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r

        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']

            assert chs * seq == x.shape[1]

            k = k.transpose(1, 2).reshape(bsz, chs, seq, dim).transpose(1, 2).flatten(0, 1)
            x = x.reshape(bsz, chs, seq, dim).transpose(1, 2).flatten(0, 1)
            # now x and k are all size of (bsz * seq, chs, dim)
            if size_old is None:
                size_old = torch.ones_like(x[..., 0:1])
            else:
                size_old = rearrange(size_old, "b (c s) d -> (b s) c d", c=chs, s=seq)

            # process it with original tome logic
            _, merge_add, _ = bipartite_soft_matching(k, self.r)

            x, size = merge_wavg_sum(merge_add, x, size_old)

            chs = x.shape[1]
            x = x.reshape(bsz, seq, chs, dim).transpose(1, 2).flatten(1, 2)
            size = rearrange(size, "(b s) c d -> b (c s) d", c=chs, s=seq)
            self.tome_container['shape'] = [bsz, chs, seq, dim]

            return x, size
        
    return ChannelToMeBlock()

def DoubleChannelToMeBlock(tome_container, r, dim):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum
    class DoubleChannelToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r

        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']

            assert chs * seq == x.shape[1]

            k = k.transpose(1, 2).reshape(bsz, chs, seq, dim).transpose(1, 2).flatten(0, 1)
            x = x.reshape(bsz, chs, seq, dim).transpose(1, 2).flatten(0, 1)
            if size_old is not None:
                size_old = rearrange(size_old, "b (c s) d -> (b s) c d", c=chs, s=seq)
            # now x and k are all size of (bsz * seq, chs, dim)

            # process it with original tome logic
            _, merge_add, _ = bipartite_soft_matching(k, self.r)

            x, size = merge_wavg_sum(merge_add, x, size_old)
            k, _ = merge_wavg_sum(merge_add, k, size_old)

            size_old = size

            _, merge_add, _ = bipartite_soft_matching(k, self.r // 2)
            x, size = merge_wavg_sum(merge_add, x, size_old)

            chs = x.shape[1]
            x = x.reshape(bsz, seq, chs, dim).transpose(1, 2).flatten(1, 2)
            size = rearrange(size, "(b s) c d -> b (c s) d", c=chs, s=seq)

            self.tome_container['shape'] = [bsz, chs, seq, dim]

            return x, size
        
    return DoubleChannelToMeBlock()

def TimestepToMeBlock(tome_container, r, dim):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum
    class TimestepToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r

        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']

            assert chs * seq == x.shape[1]

            # the shape of k is (bsz, n_head, chs*seq, dim_head)
            k = rearrange(k, "b h (c s) d -> (b c) s (h d)", c=chs, s=seq)
            x = rearrange(x, "b (c s) d -> (b c) s d", c=chs, s=seq)
            if size_old is not None:
                size_old = rearrange(size_old, "b (c s) d -> (b c) s d", c=chs, s=seq)
            # now x and k are all size of (bsz * chs, seq, dim)

            # process it with original tome logic
            _, merge_add, _ = bipartite_soft_matching(k, self.r)

            x, size = merge_wavg_sum(merge_add, x, size_old)

            seq = x.shape[1]
            self.tome_container['shape'] = [bsz, chs, seq, dim]

            x = rearrange(x, "(b c) s d -> b (c s) d", c = chs, s = seq)
            size = rearrange(size, "(b c) s d -> b (c s) d", c = chs, s = seq)
            # x = x.reshape(bsz, chs, seq, dim).flatten(1, 2)
            return x, size
        
    return TimestepToMeBlock()

def DoubleTimestepToMeBlock(tome_container, r, dim):
    from .tome.merge import bipartite_soft_matching, merge_wavg_sum
    class DoubleTimestepToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r

        def forward(self, x, k, size_old):
            bsz, chs, seq, dim = self.tome_container['shape']

            assert chs * seq == x.shape[1]

            # the shape of k is (bsz, n_head, chs*seq, dim_head)
            k = rearrange(k, "b h (c s) d -> (b c) s (h d)", c=chs, s=seq)
            x = rearrange(x, "b (c s) d -> (b c) s d", c=chs, s=seq)
            if size_old is not None:
                size_old = rearrange(size_old, "b (c s) d -> (b c) s d", c=chs, s=seq)
            # now x and k are all size of (bsz * chs, seq, dim)

            # process it with original tome logic
            _, merge_add, _ = bipartite_soft_matching(k, self.r)

            x, size = merge_wavg_sum(merge_add, x, size_old)
            k, _ = merge_wavg_sum(merge_add, k, size_old)
            
            size_old = size
            _, merge_add, _ = bipartite_soft_matching(k, self.r // 2)
            x, size = merge_wavg_sum(merge_add, x, size_old)

            seq = x.shape[1]
            self.tome_container['shape'] = [bsz, chs, seq, dim]

            x = rearrange(x, "(b c) s d -> b (c s) d", c = chs, s = seq)
            size = rearrange(size, "(b c) s d -> b (c s) d", c = chs, s = seq)
            return x, size
        
    return DoubleTimestepToMeBlock()

def RunTimeLengthToMeBlock(tome_container, r, dim):

    class RTLToMeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = r
            self.tome_container = tome_container
            self.consec = nn.Embedding(1024, dim)

        def forward(self, x : torch.Tensor, k, size_old):
            bsz, c_s, dim = x.shape
            channels = self.chs
            seq = c_s // channels

            r = min(self.r, seq - 1)
            if r <= 0:
                return x, None

            seq_new = seq - r
            k = k.transpose(1,2).reshape(bsz * channels, seq, dim)
            k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
            scores : torch.Tensor = (k[:, :-1] * k[:, 1:]).sum(-1)

            kval, kidx = torch.topk(scores, r, sorted=False)
            kidx, _ = kidx.sort(-1)

            merge_mask = torch.zeros((bsz * channels, seq), dtype=torch.bool, device=k.device)
            update_idx = kidx + 1
            merge_mask.scatter_(1, update_idx, True)
            new_indices = (~merge_mask).cumsum(dim=1) - 1
            metrix = torch.zeros((bsz * channels, seq_new, seq), device=k.device, dtype=x.dtype)
            metrix.scatter_(1, new_indices.unsqueeze(1), 1.0)

            rtl = metrix.sum(-1)
            proj = metrix / rtl.unsqueeze(-1)

            m0 = self.consec(rtl.to(torch.long)) #.view(bsz, channels, seq_new, dim)
            m0[rtl==1] = 0
            m1 = proj #.view(bsz, channels, seq_new, seq)
            m2 = x.view(bsz * channels, seq, dim)

            y = torch.baddbmm(m0, m1, m2)
            return y.view(bsz, channels * seq_new, dim), None

    return RTLToMeBlock()

# The original version
def _qk_merge0(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.exp(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.exp(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    top_v, top_idx = torch.topk(importance, seq - r)

    matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

def _qk_merge0g(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.exp(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.exp(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    matrix, _, top_idx = softtopk(similarity, importance, seq - r, 0.005)

    # matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

# The soft-label version
def _qk_merge1(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.exp(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.exp(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    top_v, top_idx = torch.topk(importance, seq - r)

    matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 0.9) # soft-labeled, we assign 0.9 to the max one, and 0.1 / (seq - r) is evenly assigned to all
    matrix_ = matrix_ + (0.1 / (seq - r))

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

# The soft-label version
def _qk_merge1g(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.exp(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.exp(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    matrix, _, top_idx = softtopk(similarity, importance, seq - r, 0.005)

    # matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 0.9) # soft-labeled, we assign 0.9 to the max one, and 0.1 / (seq - r) is evenly assigned to all
    matrix_ = matrix_ + (0.1 / (seq - r))

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

def _qk_merge2(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        thre_base = math.sqrt(8 / q.size(-1) * math.log(10))
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.sigmoid(similarity * 4 / thre_base).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        thre_base = math.sqrt(8 / q.size(-1) * math.log(10))
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.sigmoid(similarity * 4 / thre_base)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    _, top_idx = torch.topk(importance, seq - r)

    matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

def _qk_merge2g(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        thre_base = math.sqrt(8 / q.size(-1) * math.log(10))
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.sigmoid(similarity * 4 / thre_base).sum(-1)
        # importance = torch.sigmoid(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        thre_base = math.sqrt(8 / q.size(-1) * math.log(10))
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.sigmoid(similarity * 4 / thre_base)
        # similarity_ : torch.Tensor = torch.sigmoid(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    matrix, _, top_idx = softtopk(similarity, importance, seq - r, 0.005)

    # matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity

def _sst(input, thre, factor):
    # input = input
    output = torch.sigmoid(input * factor)
    o1 = torch.zeros_like(output, requires_grad=False)
    torch.where(input > thre, (1 - output).detach(), o1, out=o1)
    torch.where(input < -thre, -output.detach(), o1, out=o1)
    
    output = output + o1.detach()
    # torch
    # output[input < -thre] = (0-output)[input < -thre]
    return output

def _qk_merge4(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)

    base_thre = math.sqrt(8 / seq * math.log(10))
    similarity_ = _sst(similarity, base_thre, 4 / base_thre)
    importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1)

    # importance = torch.diagonal_scatter(similarity, torch.zeros_like(similarity[..., 0]), 0, -2, -1).sum(-1)
    top_v, top_idx = torch.topk(importance, seq - r)

    matrix = similarity_.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / (matrix.sum(-1, True) + 1e-8)

    return matrix @ x, matrix_ @ size, similarity

def _qk_merge_loss0(similarity):
    return torch.square(1 - torch.diagonal(similarity, 0, -2, -1)).mean()

def _qk_merge_loss1(similarity):
    bsz, seq, _ = similarity.shape
    loss = torch.square(similarity).sum((-2, -1)) - torch.square(torch.diagonal(similarity, 0, -2, -1)).sum(-1)
    return (seq / loss).mean()
    # return loss.mean()

def _qk_merge(r, x, q, k, size):
    x, size, _ = _qk_merge0(r, x, q, k, size)
    return x, size

def _qk_merge_grad(r, x, q, k, size):
    x, size, similarity = _qk_merge0(r, x, q, k, size)
    loss = torch.square(1 - torch.diagonal(similarity, 0, -2, -1)).mean()
    return x, size, loss

def _qk_merge_grad1(r, x, q, k, size):
    bsz, seq, dim = x.shape
    x, size, similarity = _qk_merge1(r, x, q, k, size)
    # loss = torch.square(1 - torch.diagonal(similarity, 0, -2, -1)).mean()
    # thre_base = math.sqrt(8 / q.size(-1) * math.log(10))
    # thre_scaled = thre_base #* seq / ((-torch.diagonal(similarity, 0, -2, -1) + 1) / 2).sum(-1).mean()
    # similarity_ = _sst(similarity, thre_scaled, 4 / thre_base)

    loss0 = torch.square(1 - torch.diagonal(similarity, 0, -2, -1)).mean()
    # loss1 = torch.square(similarity).sum((-2, -1)) - torch.square(torch.diagonal(similarity, 0, -2, -1)).sum(-1)
    # loss1 = (loss1 / seq / (seq - 1)).mean() * 0.01
    # loss1 = torch.square((similarity_.sum((-2, -1)) - seq * 2) / seq / seq).mean()
    return x, size, loss0 # + loss1

def _merge_grad(train, r, x, q, k, size, merge, loss):
    # bsz, seq, dim = x.shape
    merge_fn = globals().get('_qk_merge{}'.format(merge))
    if merge_fn is None:
        return x, size, 0

    # print("found merge func {}".format(merge))

    x_, size_, similarity_ = merge_fn(r, x, q, k, size)

    loss_ = 0
    if train:
        loss_c = loss
        i = 0
        while loss_c > 0:
            if loss_c & 1:
                fn = globals().get('_qk_merge_loss{}'.format(i))
                if fn is not None:
                    # print("found loss func {}".format(i))
                    loss_ = loss_ + fn(similarity_)

            loss_c >>= 1
            i += 1

    return x_, size_, loss_

# TODO: Uncompleted method
def _qk_consective_merge(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1)
    l = seq - r

    if l == 1:
        # in this case, at least one token should be reserved, and the first token is designed to be reserved
        # so we can use a lighter algorithm
        weights = q[..., 0:1, :] @ k.transpose(-1, -2)
        weights[..., 0] = 1
        weights = (weights + 1) / 2
        return weights @ x, size.sum(-2, True)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)

    similarity : torch.Tensor = q @ k.transpose(-1, -2)
    similarity_ = similarity.diagonal_scatter(torch.ones_like(similarity[..., 0]), 0, -2, -1)



    similarity_ : torch.Tensor = torch.exp(similarity)
    importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect
    # importance = torch.diagonal_scatter(similarity, torch.zeros_like(similarity[..., 0]), 0, -2, -1).sum(-1)
    top_v, top_idx = torch.topk(importance, k = l, largest=False, sorted=False)

    matrix = torch.zeros_like(importance[..., 0:1])
    matrix.scatter_(-1, top_idx, 1)


    matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size


def LearnableToMeBlock(tome_container, r, dim, q_only = False, learnable_merge = '0', learnable_loss = 0, attached = True):
    class Learnable(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, dim & -2)
            self.r = r
            self.tome_container = tome_container
            self.q_only = q_only
        
        def forward(self, x, k, size):
            if self.r <= 0:
                return x, size
            
            if size is None:
                size = torch.ones_like(x[..., 0:1])

            q_ = self.proj(x if attached else x.detach())
            # q_ = self.proj(x)
            # q_ = torch.dropout(q_, 0.5, self.training)
            if q_only:
                k_ = q_
            else:
                q_, k_ = rearrange(q_, "b s (i d) -> i b s d", i=2)

            x_, size_, loss = _merge_grad(self.training, self.r, x, q_, k_, size, learnable_merge, learnable_loss)
            self.compression_loss = loss

            return x_, size_

    return Learnable()

def LearnableChannel(tome_container, r, dim, q_only = False):
    class LearnableCH(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r
            self.proj = nn.Linear(dim, dim & -2)
            self.q_only = q_only
        
        def forward(self, x, k, size):
            if self.r <= 0:
                return x, size

            if size is None:
                size = torch.ones_like(x[..., 0:1])
            
            bsz, chs, seq, dim = self.tome_container['shape']

            x_ = rearrange(x, "b (c s) d -> (b s) c d", c=chs, s=seq)
            if self.q_only:
                q_ = k_ = self.proj(x_)
            else:
                q_, k_ = rearrange(self.proj(x_), "b s (i d) -> i b s d", i=2)
            size_ = rearrange(size, "b (c s) d -> (b s) c d", c=chs, s=seq)

            # x_, size_ = _qk_merge(self.r, x_, q_, k_, size_)
            x_, size_, loss = _qk_merge_grad(self.r, x_, q_, k_, size_)
            self.compression_loss = loss
            chs = x_.shape[1]

            self.tome_container['shape'] = bsz, chs, seq, dim
            x_ = rearrange(x_, "(b s) c d -> b (c s) d", c=chs, s=seq)
            size_ = rearrange(size_, "(b s) c d -> b (c s) d", c=chs, s=seq)

            return x_, size_
    
    return LearnableCH()

def LearnableTimestep(tome_container, r, dim, q_only = False):
    class LearnableTS(nn.Module):
        def __init__(self):
            super().__init__()
            self.tome_container = tome_container
            self.r = r
            self.proj = nn.Linear(dim, dim & -2)
            self.q_only = q_only
        
        def forward(self, x, k, size):
            if self.r <= 0:
                return x, size

            if size is None:
                size = torch.ones_like(x[..., 0:1])
            
            bsz, chs, seq, dim = self.tome_container['shape']

            x_ = rearrange(x, "b (c s) d -> (b c) s d", c=chs, s=seq)
            if self.q_only:
                q_ = k_ = self.proj(x_)
            else:
                q_, k_ = rearrange(self.proj(x_), "b s (i d) -> i b s d", i=2)
            size_ = rearrange(size, "b (c s) d -> (b c) s d", c=chs, s=seq)

            # x_, size_ = _qk_merge(self.r, x_, q_, k_, size_)
            x_, size_, loss = _qk_merge_grad(self.r, x_, q_, k_, size_)
            self.compression_loss = loss
            seq = x_.shape[1]

            self.tome_container['shape'] = bsz, chs, seq, dim
            x_ = rearrange(x_, "(b c) s d -> b (c s) d", c=chs, s=seq)
            size_ = rearrange(size_, "(b c) s d -> b (c s) d", c=chs, s=seq)

            return x_, size_
    
    return LearnableTS()

def LearnableConsectiveTimestepToMeBlock(tome_container, r, dim):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, dim & -2)

        def _compute(self, x, k_, size_old):
            bsz, seq, dim = x.shape
            q, k = rearrange(self.proj(x), "b s (i d) -> i b s d", i = 2)
            q = q / q.norm(dim=-1, keepdim=True)
            k = k / k.norm(dim=-1, keepdim=True)

            # shape of (bsz, seq)
            # Calculate the probility
            p = torch.empty_like(x[..., 0])
            p[:, 0] = 1.
            p[:, 1:] = p_ = ((q[:, :-1] * k[:, 1:]).sum(-1).neg() + 1) / 2 # in [0, 1]

            _, top_idx = torch.topk(p_, k_, sorted=False)
            b = torch.zeros_like(x[..., 0], dtype=torch.bool)
            b.scatter_(-1, top_idx + 1, True)
            b[:, 0] = True

            idx = torch.cumsum(b, -1) - 1
            matrix = torch.zeros(bsz, k_ + 1, seq, device = x.device)
            matrix.scatter_(-2, idx[..., None, :], 1)
            matrix = matrix

            c = torch.empty_like(p)

            c[b] = p[b] + (1 - p[b]).detach()
            b_ = ~b
            c[b_] = (1 - p[b_]) + p[b_].detach()

            o = (matrix / matrix.sum(-1, True)) @ (x * c[..., None])
            size = (matrix @ size_old)

            comp_g = p.mean(-1, dtype=torch.float)
            comp_f = (p > 0.5).mean(-1, dtype=torch.float)
            comp_n = seq / (seq - r)

            self.compression_loss = ((comp_n / (comp_n - 1)) * ((comp_n - 1) * comp_f * comp_g + (1 - comp_f) * (1 - comp_g))).mean() * 0.05

            return o, size
        
        def forward(self, x, k, size):
            bsz, chs, seq, dim = tome_container['shape']

            x_ = rearrange(x, "b (c s) d -> (b c) s d", c=chs, s=seq)
            if size is None:
                size_ = torch.ones_like(x_[..., 0, None])
            else:
                size_ = rearrange(size, "b (c s) d -> (b c) s d", c=chs, s=seq)

            r_ = min(max(r, 0), seq - 1)
            k_ = seq - 1 - r_
            x_, size_ = self._compute(x_, k_, size_)

            seq = x_.shape[1]
            tome_container['shape'] = (bsz, chs, seq, dim)

            x_ = rearrange(x_, "(b c) s d -> b (c s) d", c=chs, s=seq)
            size_ = rearrange(size_, "(b c) s d -> b (c s) d", c=chs, s=seq)

            return x_, size_

    return Block()
