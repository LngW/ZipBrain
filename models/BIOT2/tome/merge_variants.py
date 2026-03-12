
from typing import Callable, Tuple

import torch
from einops import rearrange

from tome.merge import merge_source

def apply_merge(pinfo, r : int, variant : str, x, metric):

    if variant is None or variant == '' or variant.startswith('tome'):
        return tome_merge(pinfo, r, x, metric)
    elif variant.startswith('full'):
        return full_merge(pinfo, r, x, metric)
    # elif variant.startswith('fch'):
    #     return fch_merge(pinfo, r, x, metric)
    elif variant.startswith('meann'):
        return mean_merge(pinfo, r, x, metric, True)
    elif variant.startswith('mean'):
        return mean_merge(pinfo, r, x, metric, False)
    elif variant.startswith('clsp'):
        return cls_prune(pinfo, r, x, metric)
    elif variant.startswith('clsm'):
        return cls_merge(pinfo, r, x, metric)
    elif variant.startswith('dartm'):
        return dart_merge(pinfo, r, x, metric)
    elif variant.startswith('dartp'):
        return dart_prune(pinfo, r, x, metric)

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

    # if soft_hard_mode == 1:
    #     return hard_merge, hard_merge
    # elif soft_hard_mode == 2:
    #     return soft_merge, soft_merge
    # else:
    return soft_merge, hard_merge, similarity

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

    pinfo['attn_score'] = scores

    if pinfo["trace_source"]:
        pinfo["source"] = merge_source(
            merge, x, pinfo["source"]
        )
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

    score = (m_r @ token_mean).squeeze(-1)
    vals, sort_indicies = score.sort(dim=-1)

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
    left_indicies, _ = left_indicies.sort()
    left_indicies = left_indicies[..., r:]


    def merge(x, mode='mean'):
        dim = x.size(-1)
        x_prot, x_raw = x[slice_p], x[slice_r]

        src = torch.gather(x_raw, -2, src_idx[..., None].expand(-1, -1, dim))
        x_raw = torch.scatter_reduce(x_raw, -2, tar_idx[..., None].expand(-1, -1, dim), src, mode)
        x_raw = torch.gather(x_raw, -2, left_indicies[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, x_raw], dim=-2)
    
    return merge, merge, score

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
    # non_pivot_idx = torch.arange(0, seq, 1)[None, ...].repeat(bsz, 1)
    # non_pivot_idx = non_pivot_idx.scatter_(-1, pivot_idx, 0)
    # non_pivot_idx = non_pivot_idx.sort().indices[..., k:]
    # non_pivot_tokens = metric.gather(-2, non_pivot_idx[..., None].expand(-1, -1, dim))

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
