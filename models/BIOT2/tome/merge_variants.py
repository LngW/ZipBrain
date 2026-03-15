
import math
from typing import Callable, Tuple

import torch
from einops import rearrange

from tome.merge import merge_source

Spliter = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

def apply_merge(pinfo, r : int, variant : str, x, metric):

    protected = 0
    if pinfo['class_token']:
        protected += 1
    if pinfo['distill_token']:
        protected += 1
    t = metric.size(1) - protected
    r_ = min(r, t - 1)
    if r_ <= 0:
        return x
    
    def split(input):
        return input[..., :protected, :], input[..., protected:, :]

    if variant is None or variant == '' or variant.startswith('tome'):
        return tome_merge(pinfo, r, x, metric)
    elif variant.startswith('full'):
        return full_merge(pinfo, r, x, metric)
    elif variant.startswith('rpe'):
        return rpe_merge(pinfo, r, x, metric)
    elif variant.startswith('alibi'):
        return alibi_merge(pinfo, r, x, metric, split)
    # elif variant.startswith('fch'):
    #     return fch_merge(pinfo, r, x, metric)
    elif variant.startswith('meann'):
        return mean_merge(pinfo, r, x, metric, True)
    elif variant.startswith('meanm'):
        return mean_merge(pinfo, r, x, metric, False)
    elif variant.startswith('meanp'):
        return mean_prune(pinfo, r_, x, metric, split)
    elif variant.startswith('clsp'):
        return cls_prune(pinfo, r, x, metric)
    elif variant.startswith('clsm'):
        return cls_merge(pinfo, r, x, metric)
    elif variant.startswith('dartm'):
        return dart_merge(pinfo, r, x, metric)
    elif variant.startswith('dartp'):
        return dart_prune(pinfo, r, x, metric)
    elif variant.startswith('rndm'):
        return random_merge(pinfo, r_, x, metric, split)
    elif variant.startswith('rndp'):
        return random_prune(pinfo, r_, x, metric, split)

    return x

# =============== #
#     helpers     #
# =============== #

def handle_source(pinfo, x, merge):
    if not pinfo['trace_source']:
        return
    
    source = pinfo['source']
    if source is None:
        bsz, seq, _ = x.shape
        source = torch.eye(seq, device=x.device)[None, ...].repeat(bsz, 1, 1)

    pinfo['source'] = merge(source)

def handle_size(pinfo, x, merge):
    size = pinfo['size']
    if size is None:
        size = torch.ones_like(x[..., 0:1])

    size_ = merge(size)
    pinfo['size'] = size_
    return size, size_

def do_nothing(x, mode = 'ignored'):
    return x

# =============== #
# implementations #
# =============== #

def tome_merge(pinfo, r, x, metric):

    if r <= 0:
        return x

    from tome.merge import bipartite_soft_matching, merge_wavg
    merge, _, scores = bipartite_soft_matching(
        metric,
        r,
        pinfo["class_token"],
        pinfo["distill_token"],
    )

    pinfo['attn_score'] = scores

    if pinfo["trace_source"]:
        pinfo["source"] = merge_source(
            merge, x, pinfo["source"]
        )
    x, pinfo["size"] = merge_wavg(
        merge, x, pinfo["size"]
    )

    return x

def random_merge(pinfo, r, x, metric, split : Spliter):
    m_prot, m_raw = split(metric)

    with torch.no_grad():
        bsz, seq, dim = m_raw.shape
        rnd = torch.Generator(m_raw.device)

        indices = torch.stack([torch.randperm(seq, generator=rnd, device=m_raw.device) for _ in range(bsz)])
        # indices = torch.randperm(bsz * seq, generator=rnd, device=m_raw.device)
        # indices = indices.view(bsz, seq) - (torch.arange(0, bsz, device=m_raw.device) * seq).view(bsz, 1)

        src_idx = indices[:, :r, None]
        left_idx = indices[:, r:, None]
        tar_idx = torch.randint(0, seq-r, (bsz, r, 1), device=m_raw.device)

    def merge(x, mode = 'sum'):
        x_prot, x_raw = split(x)

        dim = x_raw.size(-1)
        src = x_raw.gather(-2, src_idx.expand(-1, -1, dim))
        left = x_raw.gather(-2, left_idx.expand(-1, -1, dim))

        left = left.scatter_reduce(-2, tar_idx.expand(-1, -1, dim), src, mode)
        
        return torch.cat([x_prot, left], -2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def random_prune(pinfo, r, x, metric, split : Spliter):
    m_prot, m_raw = split(metric)

    with torch.no_grad():
        bsz, seq, dim = m_raw.shape
        rnd = torch.Generator(m_raw.device)

        indices = torch.stack([torch.randperm(seq, generator=rnd, device=m_raw.device) for _ in range(bsz)])

        # indices = torch.randperm(bsz * seq, generator=rnd, device=m_raw.device)
        # indices = indices.view(bsz, seq) - (torch.arange(0, bsz, device=m_raw.device) * seq).view(bsz, 1)
        left_idx = indices[:, r:, None]

    def merge(x, mode = 'sum'):
        x_prot, x_raw = split(x)

        dim = x_raw.size(-1)
        left = x_raw.gather(-2, left_idx.expand(-1, -1, dim))

        # left = left.scatter_reduce(-2, tar_idx.expand(-1, -1, dim), src, mode)
        
        return torch.cat([x_prot, left], -2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def _full_merge(x, metric, r, class_token, distill_token):
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    t = metric.shape[1]
    t_ = t - protected
    r = min(r, t_ - 1)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        q = metric[..., protected:, :]
        q = q / q.norm(dim = -1, keepdim = True)

        similarity = q @ q.transpose(-2, -1)
        importance = torch.exp(similarity).sum(-1)

        _, top_idx = torch.topk(importance, t_ - r)

        matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, t - protected))
        matrix = (matrix + 1) / 2
        matrix.scatter_(-1, top_idx[..., None], 1)

        matrix_ = torch.zeros_like(matrix)
        matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

        # similarity = None
        importance = None
        top_idx = None

        matrix = matrix * matrix_
        # matrix_ = matrix_ / matrix_.sum(-1, True)
        # matrix = matrix / matrix.sum(-1, True)

    def soft_merge(x : torch.Tensor, mode = 'ignored'):
        right = matrix @ x[..., protected:, :]
        if mode == 'mean':
            right = right / matrix.sum(-1, True)
        
        return torch.cat([x[..., :protected, :], right], dim=-2)

    def hard_merge(x : torch.Tensor, mode = 'ignored'):
        right = matrix_ @ x[..., protected:, :]
        if mode == 'mean':
            right = right / matrix_.sum(-1, True)

        return torch.cat([x[..., :protected, :], right], dim=-2)
        # return torch.cat([x[..., :protected, :], matrix_ @ x[..., protected:, :]], dim=-2)

    return soft_merge, hard_merge, similarity

# A bad implementation. 
# After once two token merged, how to measure the distance between this merged token with other tokens?
# Those token may not event in same channel or at same timestep
def alibi_merge(pinfo, r, x, metric, split : Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]):
    with torch.no_grad():
        alibi = pinfo.get('alibi', None)
        if alibi is None:
            bsz, chs, seq, dim = pinfo['shape']
            alibi = torch.arange(seq, dtype=metric.dtype, device=metric.device).view(1, -1)
            alibi = (alibi - alibi.view(-1, 1)).abs() / seq
            alibi = alibi.unsqueeze(0).repeat(bsz, chs, chs)
        else:
            alibi : torch.Tensor = alibi

        m_prot, m_raw = split(metric)

        m_raw = m_raw / m_raw.norm(p=2, dim=-1, keepdim=True)
        similarity = m_raw @ m_raw.transpose(-2, -1)
        similarity = similarity - alibi

        importance = torch.exp(similarity).sum(-1)
        left_idx = importance.topk(m_raw.size(1) - r, dim=-1, sorted=False).indices.sort().values

        left_similarity = similarity.gather(-2, left_idx[..., None].expand(-1, -1, similarity.size(-1)))
        max_idx = left_similarity.argmax(-2, True)
        matrix_hard = torch.zeros_like(left_similarity).scatter(-2, max_idx, 1)
        left_similarity = None

    def merge_hard(input):
        i_prot, i_raw = split(input)
        i_raw = matrix_hard @ i_raw
        return torch.cat([i_prot, i_raw], dim=-2)
    
    handle_source(pinfo, x, merge_hard)
    size, size_ = handle_size(pinfo, x, merge_hard)

    # alibi = (matrix_hard @ (alibi * size)).gather(-1, left_idx[..., None, :].expand(-1, left_idx.size(-1), -1))
    alibi = alibi.gather(-2, left_idx[..., None].expand(-1, -1, left_idx.size(-1))).contiguous()
    torch.diagonal_scatter(alibi, torch.diagonal(alibi, 0, -2, -1) - torch.diagonal(alibi, 0, -2, -1), 0, -2, -1)
    # alibi.fill_diagonal_(0)
    # alibi = alibi.gather(-1, left_idx[..., None, :].expand(-1, left_idx.size(-1), -1)).contiguous()
    # alibi = alibi / size_
    pinfo['alibi'] = alibi

    pinfo['attn_score'] = similarity

    return merge_hard(x * size) / size_

def full_merge(pinfo, r, x, metric):

    protected = 0
    if pinfo['class_token']:
        protected += 1
    if pinfo['distill_token']:
        protected += 1
    bsz, seq, dim = metric.shape
    t = seq - protected
    r = min(r, t - 1)

    if r <= 0:
        return x
    
    def split(input):
        return input[..., :protected, :], input[..., protected:, :]

    with torch.no_grad():
        q = metric[..., protected:, :]
        q = q / q.norm(dim = -1, keepdim = True)

        similarity = q @ q.transpose(-2, -1)
        importance = torch.exp(similarity).sum(-1)

        top_idx = torch.topk(importance, t - r).indices.sort().values

        matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, t - protected))
        matrix = (matrix + 1) / 2
        matrix.scatter_(-1, top_idx[..., None], 1)

        matrix_ = torch.zeros_like(matrix)
        matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

        # similarity = None
        importance = None
        top_idx = None

        matrix = matrix * matrix_
    
    def merge(x, mean = False):
        x_prot, x_raw = split(x)
        x_raw = matrix_ @ x_raw
        if mean:
            x_raw = x_raw / matrix_.sum(-1, True)

        return torch.cat([x_prot, x_raw], dim=-2)
    
    if pinfo['trace_source']:
        pinfo['attn_score'] = similarity

    handle_source(pinfo, x, lambda it : merge(it, False))
    handle_size(pinfo, x, lambda it : merge(it, True))

    x_prot, x_raw = split(x)
    x_raw = matrix @ x_raw / matrix.sum(-1, True)
    return torch.cat([x_prot, x_raw], dim = -2)

    # return x_

def rpe_merge(pinfo, r, x, metric):

    protected = 0
    if pinfo['class_token']:
        protected += 1
    if pinfo['distill_token']:
        protected += 1
    bsz, seq, dim = metric.shape
    t = seq - protected
    r = min(r, t - 1)

    if r <= 0:
        return x
    
    def split(input):
        return input[..., :protected, :], input[..., protected:, :]
    
    with torch.no_grad():
        pe_score = pinfo['pe_score']
        if pe_score is None:
            _, chs, seq_, _ = pinfo['shape']
            pe = pinfo['pe'][:, :seq_, :]
            pe_score = pe @ pe.transpose(-2, -1)

            pe_score = (pe_score - pe_score.min()) / (pe_score.amax() - pe_score.amin())
            pe_score = pe_score.to(x.device)
            pe_score = pe_score / math.sqrt(2)
            pe_score = pe_score.repeat(bsz, chs, chs)

        pe_score = pe_score / math.sqrt(2)
        # pinfo['pe_score'] = pe_score

        q = metric[..., protected:, :]
        q = q / q.norm(dim = -1, keepdim = True)

        similarity = q @ q.transpose(-2, -1)
        bsz, seq, _ = similarity.shape
        similarity_ = similarity.view(bsz, -1)
        pe_score_ = pe_score.view(bsz, -1)
        # print(similarity_.device)
        # print(pe_score.device)
        # print(pe_score_.device)
        pe_score_ = pe_score_ * (similarity_.amax(-1, True) - similarity_.amin(-1, True))
        pe_score_ = pe_score_ + similarity_.amin(-1, True)
        pe_score_ = pe_score_.view(bsz, seq, seq)
        similarity = similarity - pe_score_
        importance = torch.exp(similarity).sum(-1)

        top_idx = torch.topk(importance, t - r).indices.sort().values

        matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, t - protected))
        matrix = (matrix + 1) / 2
        matrix.scatter_(-1, top_idx[..., None], 1)

        matrix_ = torch.zeros_like(matrix)
        matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

        # similarity = None
        importance = None
        # top_idx = None

        matrix = matrix * matrix_
    
    def merge(x, mean = False):
        x_prot, x_raw = split(x)
        x_raw = matrix_ @ x_raw
        if mean:
            x_raw = x_raw / matrix_.sum(-1, True)

        return torch.cat([x_prot, x_raw], dim=-2)
    
    if pinfo['trace_source']:
        pinfo['attn_score'] = similarity

    handle_source(pinfo, x, lambda it : merge(it, False))
    handle_size(pinfo, x, lambda it : merge(it, True))

    x_prot, x_raw = split(x)
    x_raw = matrix @ x_raw / matrix.sum(-1, True)

    # pe_score = 
    pinfo['pe_score'] = (matrix_ @ pe_score).gather(dim=-1, index=top_idx[..., None, :].expand(-1, seq - r, -1))

    return torch.cat([x_prot, x_raw], dim = -2)

    # return x_

# def fch_merge(pinfo, r, x, metric):
#     bsz, chs, seq, dim = pinfo['shape']
#     def reshaper(input):
#         return rearrange(input, "b (c s) d -> (b s) c d", s=seq)
    
#     r = min(r, chs - 1)
#     # print(r)
#     if r <= 0:
#         return x
    
#     protected = 0
#     if pinfo['class_token']:
#         protected += 1
#     if pinfo['distill_token']:
#         protected += 1

#     def handle_protected(input):
#         prot = input[..., :protected, :]
#         raw = input[..., protected:, :]

#         raw = reshaper(raw)
#         return prot, raw

#     x_prot, x_raw = handle_protected(x)
#     m_prot, m_raw = handle_protected(metric)

#     size = pinfo['size']
#     if size is None:
#         size = torch.ones_like(x[..., 0:1])
#     size_prot, size_raw = handle_protected(size)
    
#     if pinfo['trace_source']:
#         source = pinfo['source']
#         if source is None:
#             source = torch.eye(chs * seq)[None, ...].repeat(bsz, 1, 1)
#         source_prot, source_raw = handle_protected(source)

#     del reshaper, handle_protected
#     def reshaper(input):
#         return rearrange(input, "(b s) c d -> b (c s) d", s = seq)
    
#     def handle_protected(prot, raw):
#         raw = reshaper(raw)
#         return torch.cat([prot, raw], dim = -2)

#     soft, hard = _full_merge(x_raw, m_raw, r, False, False)

#     if pinfo['trace_source']:
#         source_raw = hard(source_raw)
#         pinfo['source'] = handle_protected(source_prot, source_raw)
    
#     x_raw, size_raw = merge_wavg(soft, hard, x_raw, size_raw)
#     chs = x_raw.size(-2)
#     x = handle_protected(x_prot, x_raw)
#     size = handle_protected(size_prot, size_raw)

#     pinfo['size'] = size
#     pinfo['shape'] = bsz, chs, seq, dim
#     return x

def mean_merge(pinfo, r, x, metric, centerize):
    if r <= 0: 
        return x
    from tome.merge import merge_source, merge_wavg
    merge, _, scores = _mean_merge(
        pinfo,
        r,
        x,
        metric,
        centerize
    )

    if pinfo["trace_source"]:
        pinfo["source"] = merge_source(
            merge, x, pinfo["source"]
        )
        pinfo['attn_score'] = scores

    x, pinfo["size"] = merge_wavg(
        merge, x, pinfo["size"]
    )

    return x

def _mean_merge(pinfo, r, x, metric, centerize):
    cls = pinfo["class_token"]
    dst = pinfo["distill_token"]

    protected = 0
    if cls:
        protected += 1
    if dst:
        protected += 1

    r = min(r, x.shape[1] - protected - 1)

    if r <= 0:
        return do_nothing, do_nothing
    
    slice_p = (..., slice(None, protected), slice(None, None))
    slice_r = (..., slice(protected, None), slice(None, None))

    # x_p, x_r = x[slice_p], x[slice_r]
    m_p, m_r = metric[slice_p], metric[slice_r]

    token_mean : torch.Tensor = m_r.mean(-2, True)
    if centerize:
        m_r = m_r - token_mean
    token_mean = token_mean / token_mean.norm(2, -1, True)
    token_mean = token_mean.transpose(-2, -1)
    m_r = m_r / m_r.norm(2, -1, True)

    score = (m_r @ token_mean).squeeze(-1) # (bsz, seq)
    vals, sort_indicies = score.sort(dim=-1) # (bsz, seq)

    diff = vals[..., 1:] - vals[..., :-1] # (bsz, seq - 1)
    _, top_indicies = torch.topk(diff, r, largest=False) # (bsz, r)

    top_mask = torch.zeros_like(sort_indicies[..., 1:]) # (bsz, seq - 1)
    top_mask.scatter_(-1, top_indicies, 1)
    top_shift = top_mask.cumsum(-1)
    top_counts = top_shift - (top_shift * (1 - top_mask)).cummax(-1).values
    top_counts = top_counts.gather(-1, top_indicies) # (bsz, seq - 1)

    src_idx = sort_indicies.gather(-1, top_indicies + 1)
    tar_idx = sort_indicies.gather(-1, top_indicies - top_counts + 1)

    bsz, seq = sort_indicies.shape
    left_mask = torch.ones_like(sort_indicies)
    left_mask.scatter_(-1, src_idx, 0)
    left_indicies = torch.arange(0, sort_indicies.size(-1), 1, device=x.device)
    left_indicies = torch.masked_select(left_indicies, left_mask.to(dtype=torch.bool))
    left_indicies = left_indicies.view(bsz, seq - r)
    # left_indicies = left_indicies[None, ...] * left_mask
    # left_indicies, _ = left_indicies.sort()
    # left_indicies = left_indicies[..., r:]

    def merge(x, mode='mean'):
        dim = x.size(-1)
        x_prot, x_raw = x[slice_p], x[slice_r]

        src = torch.gather(x_raw, -2, src_idx[..., None].expand(-1, -1, dim))
        x_raw = torch.scatter_reduce(x_raw, -2, tar_idx[..., None].expand(-1, -1, dim), src, mode)
        x_raw = torch.gather(x_raw, -2, left_indicies[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, x_raw], dim=-2)
    
    return merge, merge, score

def mean_merge1(pinfo, r, x, metric, split : Spliter):
    m_prot, m_raw = split(metric)

    with torch.no_grad():
        bsz, seq, dim = m_raw.shape

        mean = m_raw.mean(-2, True) # (bsz, 1, dim)
        m_raw = m_raw - mean
        m_raw = m_raw / m_raw.norm(2, -1, True) # (bsz, seq, dim)

        scores = (m_raw @ mean.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
        sort_val, sort_idx = scores.sort()


def mean_prune(pinfo, r, x, metric, split : Spliter):
    m_prot, m_raw = split(metric)

    with torch.no_grad():
        bsz, seq, dim = m_raw.shape

        mean = m_raw.mean(-2, True) # (bsz, 1, dim)
        m_raw = m_raw - mean
        scores = ((m_raw / m_raw.norm(2, -1, True)) @ mean.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq, dim) @ (bsz, dim, 1) -> (bsz, seq, 1) -> (bsz, seq)

        # scores.sort
        mask = torch.ones_like(scores, dtype=torch.bool) # (bsz, seq)
        mask.scatter_(-1, scores.topk(r, -1, False).indices, False)

        def merge(x):
            x_p, x_r = split(x)
            x_r = x_r[mask]
            x_r = x_r.view(bsz, -1, x_r.size(-1))

            return torch.cat([x_p, x_r], dim=-2)
        
        handle_source(pinfo, x, merge)
        size, size_ = handle_size(pinfo, x, merge)

        pinfo['attn_score'] = scores

        return merge(x)

def cls_prune(pinfo, r, x, metric):
    if not pinfo['class_token']:
        return x
    
    protected = 1
    if pinfo['distill_token']:
        protected += 1
    def split_prot_raw(input):
        return input[..., :protected, :], input[..., protected:, :]

    bsz, seq, dim = metric.shape
    r = min(r, seq - protected - 1)
    t = seq - protected - r

    if r <= 0:
        return x

    # x_prot, x_raw = split_prot_raw(x)
    m_prot, m_raw = split_prot_raw(metric)

    cls_token = metric[..., 0:1, :]
    cls_attn = cls_token @ m_raw.transpose(-2, -1)
    _, top_idx = torch.topk(cls_attn, t, dim=-1, sorted=False)

    top_idx, _ = top_idx.sort()
    top_idx = top_idx.squeeze(-2).unsqueeze(-1)

    def merge(it):
        it_prot, it_raw = split_prot_raw(it)
        # dim = it.size(-1)
        it_raw = torch.gather(it_raw, -2, top_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, it_raw], dim=-2)
    
    handle_size(pinfo, x, merge)
    handle_source(pinfo, x, merge)
    pinfo['attn_score'] = cls_attn

    return merge(x)

def cls_merge(pinfo, r, x, metric):
    if not pinfo['class_token']:
        return x
    
    protected = 1
    if pinfo['distill_token']:
        protected += 1
    def split_prot_raw(input):
        return input[..., :protected, :], input[..., protected:, :]

    bsz, seq, dim = metric.shape
    r = min(r, seq - protected - 1)
    t = seq - protected - r

    if r <= 0:
        return x

    # x_prot, x_raw = split_prot_raw(x)
    m_prot, m_raw = split_prot_raw(metric)

    cls_token = metric[..., 0:1, :]
    cls_attn = (cls_token @ m_raw.transpose(-2, -1)).squeeze(-2)
    pinfo['attn_score'] = cls_attn

    # score = (m_r @ token_mean).squeeze(-1)
    vals, sort_indicies = cls_attn.sort(dim=-1)

    diff = vals[..., 1:] - vals[..., :-1]
    _, top_indicies = torch.topk(diff, r, largest=False)

    top_mask = torch.zeros_like(sort_indicies[..., 1:])
    top_mask.scatter_(-1, top_indicies, 1)
    top_shift = top_mask.cumsum(-1)
    top_counts = top_shift - (top_shift * (1 - top_mask)).cummax(-1).values
    top_counts = top_counts.gather(-1, top_indicies)

    src_idx = sort_indicies.gather(-1, top_indicies + 1)
    tar_idx = sort_indicies.gather(-1, top_indicies - top_counts + 1)

    left_mask = torch.ones_like(sort_indicies)
    left_mask.scatter_(-1, src_idx, 0)
    left_indicies = torch.arange(0, sort_indicies.size(-1), 1, device=x.device)
    left_indicies = left_indicies[None, ...] * left_mask
    # left_indicies = (left_mask.cumsum(-1) - 1) * left_mask
    left_indicies, _ = left_indicies.sort()
    left_indicies = left_indicies[..., r:]

    def merge(x, mode='mean'):
        dim = x.size(-1)
        x_prot, x_raw = split_prot_raw(x)

        src = torch.gather(x_raw, -2, src_idx[..., None].expand(-1, -1, dim))
        x_raw = torch.scatter_reduce(x_raw, -2, tar_idx[..., None].expand(-1, -1, dim), src, mode)
        x_raw = torch.gather(x_raw, -2, left_indicies[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, x_raw], dim=-2)

    size, size_ = handle_size(pinfo, x, merge)
    handle_source(pinfo, x, merge)

    return merge(x * size) / size_.sum(-1, True)
    

def dart_merge(pinfo, r, x, metric : torch.Tensor):
    protected = 0
    if pinfo['class_token']:
        protected += 1
    if pinfo['distill_token']:
        protected += 1
    
    def split(input):
        return input[..., :protected, :], input[..., protected:, :]
    
    bsz, seq, dim = metric.shape
    r = min(r, seq - protected - 1)
    if r <= 0:
        return x
    
    seq = seq - protected
    _, metric = split(metric)

    # select ~5% tokens as pivot, at least one token
    k = seq * 5 // 100 + 1
    # select tokens via metric's l1-norm
    pivot_idx = metric.norm(dim=-1, p=1).topk(k=k, dim=-1).indices
    pivot_tokens = metric.gather(dim=-2, index=pivot_idx[..., None].expand(-1, -1, dim))

    similarity = (pivot_tokens / pivot_tokens.norm(2, -1, True)) @ (metric / metric.norm(2, -1, True)).transpose(-2, -1)

    sim_max_val, sim_max_idx = similarity.max(dim=-2)
    _, src_idx = sim_max_val.scatter(-1, pivot_idx, -1).topk(r, -1)

    # similarity = None

    tar_idx = sim_max_idx.gather(-1, src_idx)
    tar_idx = pivot_idx.gather(-1, tar_idx)

    # pivot_idx = None

    left_idx = torch.arange(0, seq, device=metric.device)[None, ...]
    left_idx = left_idx.repeat(bsz, 1)
    left_idx = left_idx.scatter(-1, src_idx, 0)
    left_idx = left_idx.sort().values[..., r:]

    def merge(x:torch.Tensor):
        x_prot, x_raw = split(x)

        dim = x.size(-1)
        src = x_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        tar = x_raw.scatter_add(-2, tar_idx[..., None].expand(-1, -1, dim), src)

        x_raw = tar.gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, x_raw], dim = -2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)
    x = merge(x * size) / size_

    return x

def dart_prune(pinfo, r, x, metric : torch.Tensor):
    protected = 0
    if pinfo['class_token']:
        protected += 1
    if pinfo['distill_token']:
        protected += 1
    
    def split(input):
        return input[..., :protected, :], input[..., protected:, :]
    
    bsz, seq, dim = metric.shape
    r = min(r, seq - protected - 1)
    if r <= 0:
        return x
    
    seq = seq - protected
    _, metric = split(metric)

    # select ~5% tokens as pivot, at least one token
    k = seq * 5 // 100 + 1
    # select tokens via metric's l1-norm
    pivot_idx = metric.norm(dim=-1, p=1).topk(k=k, dim=-1).indices
    pivot_tokens = metric.gather(dim=-2, index=pivot_idx[..., None].expand(-1, -1, dim))

    similarity = (pivot_tokens / pivot_tokens.norm(2, -1, True)) @ (metric / metric.norm(2, -1, True)).transpose(-2, -1)
    sim_max_val, sim_max_idx = similarity.max(dim=-2)
    _, src_idx = sim_max_val.scatter(-1, pivot_idx, -1).topk(r, -1)

    tar_idx = sim_max_idx.gather(-1, src_idx)
    tar_idx = pivot_idx.gather(-1, tar_idx)

    left_idx = torch.arange(0, seq, device=metric.device)[None, ...]
    left_idx = left_idx.repeat(bsz, 1)
    left_idx = left_idx.scatter(-1, src_idx, 0)
    left_idx = left_idx.sort().values[..., r:]

    def merge(x:torch.Tensor):
        x_prot, x_raw = split(x)

        dim = x.size(-1)
        # src = x_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        # tar = x_raw.scatter_add(-2, tar_idx[..., None].expand(-1, -1, dim), src)

        x_raw = x_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, x_raw], dim = -2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)
    x = merge(x * size) / size_

    return x
