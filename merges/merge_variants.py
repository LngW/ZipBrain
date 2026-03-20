
import math
from typing import Callable, Tuple

import torch
from einops import rearrange

# from tome.merge import merge_source

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
    elif variant.startswith('kiddp'): # Keep Important Drop Duplicated, merge to pivot
        return kidd_pivot(pinfo, r_, x, metric, split)
    # elif variant.startswith('dartd'): # DART Combined
    #     return merge_prune1(pinfo, r_, x, metric, split)
    elif variant.startswith('kiddl'): # Keep Important Drop Duplicated, merge to left
        return kidd_left(pinfo, r_, x, metric, split)

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

    from tome.merge import bipartite_soft_matching, merge_wavg, merge_source
    merge, _ = bipartite_soft_matching(
        metric,
        r,
        pinfo["class_token"],
        pinfo["distill_token"],
    )

    # pinfo['attn_score'] = scores

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

# drop un-important and duplicate tokens, merge importance but duplicate tokens
def merge_prune(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape

    imp_num = seq - r
    pivot_num = math.ceil((seq - r) * 0.05)
    # indices = torch.arange(0, seq, device=metric.device).view(1, seq).expand(bsz, -1)
    
    mean_token = m_raw.mean(-2, True) # (bsz, 1, dim)
    # mean_norm =  # (bsz, 1, 1)
    mean_token = mean_token / mean_token.norm(2, -1, True)

    metric_norm = m_raw.norm(2, -1, True) # (bsz, seq, 1)
    m_raw = m_raw / metric_norm
    imp_score = (m_raw @ mean_token.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
    imp_idx = imp_score.topk(imp_num, -1, True, False).indices

    # pivot_num = math.ceil(math.log(seq - r))

    # pivot_num = math.ceil(seq / 20)
    # pivot_idx = metric_norm.squeeze(-1).gather(-1, imp_idx).topk(pivot_num, -1, True, False).indices # (bsz, p)
    tmp_idx = metric_norm.squeeze(-1).sort(-1, descending=True).indices # (bsz, p)
    pivot_idx = tmp_idx[:, :pivot_num] # (bsz, p)
    # non_pivot_idx = tmp_idx[:, pivot_num:] # (bsz, seq - p)

    # non_pivot_idx = indices[
    #     torch.ones_like(indices, dtype=torch.bool).scatter_(-1, pivot_idx, False)
    # ].view(bsz, -1) # (bsz, seq - p)

    pivots = m_raw.gather(-2, pivot_idx[..., None].expand(-1, -1, dim))
    # non_pivots = m_raw.gather(-2, non_pivot_idx[..., None].expand(-1, -1, dim))

    # non_pivot_non_imp_num = min(seq - pivot_num, r)
    sim_score, sim_idx = (m_raw @ pivots.transpose(-2, -1)).max(-1) # (bsz, seq)
    sim_score = sim_score.scatter(-1, pivot_idx, -2)

    tmp_idx = sim_score.sort(-1, descending=True).indices
    dup_idx = tmp_idx[:, :r]
    left_idx = tmp_idx[:, r:]

    tar_idx = pivot_idx.gather(-1, sim_idx).gather(-1, dup_idx)
    src_idx = dup_idx

    # non_dup_idx = non_pivot_idx.gather(-1, non_dup_idx)

    imp_mask = torch.zeros_like(m_raw[:, :, 0], dtype=torch.bool).scatter_(-1, imp_idx, True)
    imp_dup_mask = imp_mask.gather(-1, dup_idx)

    # imp_dup_num = min(imp_num, r)
    # imp_dup_value, imp_dup_idx = imp_dup_mask.to(dtype=torch.int).topk(imp_dup_num, -1, sorted=False) # (bsz, k)

    # tar_idx = tar_idx.gather(-1, imp_dup_idx)
    # src_idx = dup_idx.gather(-1, imp_dup_idx)

    # tar_idx[imp_dup_value != 1] = -1
    # src_idx[imp_dup_value != 1] = -1
    tar_idx[~imp_dup_mask] = -1
    src_idx[~imp_dup_mask] = -1

    tar_idx = tar_idx + 1
    src_idx = src_idx + 1

    # tar_idx = torch.where(imp_dup_value == 1, sim_idx.gather(-1, imp_dup_idx), -1) + 1
    # src_idx = torch.where(imp_dup_value == 1, dup_idx.gather(-1, imp_dup_idx), -1) + 1
    # left_idx = indices.scatter(-1, dup_idx, -1).sort(-1).values[:, r:]
    # assert left_idx.size(1) == seq - r
    # left_idx = torch.cat([pivot_idx, non_dup_idx], dim=-1)
    left_idx = left_idx.sort().values
    assert left_idx.size(1) == seq - r

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape
        x_raw_ = torch.cat([torch.zeros_like(x_raw[:, 0:1, :]), x_raw], dim = -2)
        src = x_raw_.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        x_raw_ = x_raw_.scatter_reduce(-2, tar_idx[..., None].expand(-1, -1, dim), src, reduce)
        left = x_raw_[:, 1:].gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    # slerp
    # size, size_ = handle_size(pinfo, x, merge)
    # return merge(x * size) / size_

    # mlerp
    length = x.norm(2, -1, True)
    length_ = merge(length, 'amax')
    x = merge(x)
    x = x / x.norm(2, -1, True)
    x = x * length_
    return x

    # avg
    # size = torch.ones_like(x[..., :1])
    # size_ = merge(size)
    # return merge(x) / merge(size)

    # length_avg
    # length = x.norm(2, -1, True)
    # size, size_ = handle_size(pinfo, x, merge)
    # length_ = merge(length * size) / size_
    # x = merge(x)
    # x = x / x.norm(2, -1, True) * length_
    # return x

# Merge tokens with similar mean score, rather than with the pivot token.
def merge_prune1(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape
    t = seq - r

    indices = torch.arange(0, seq, device=metric.device).view(1, seq).expand(bsz, -1)
    
    mean_token = m_raw.mean(-2, True) # (bsz, 1, dim)
    mean_norm = mean_token.norm(2, -1, True) # (bsz, 1, 1)
    mean_token = mean_token / mean_norm

    metric_norm = m_raw.norm(2, -1, True) # (bsz, seq, 1)
    m_raw = m_raw / metric_norm
    imp_score = (m_raw @ mean_token.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
    imp_idx = imp_score.topk(t, -1, True, False).indices

    pivot_num = math.ceil(seq * 0.05)
    pivot_idx = metric_norm.squeeze(-1).topk(pivot_num, -1, True, False).indices # (bsz, p)
    non_pivot_idx = indices[
        torch.ones_like(indices, dtype=torch.bool).scatter_(-1, pivot_idx, False)
    ].view(bsz, -1) # (bsz, seq - p)

    pivots = m_raw.gather(-2, pivot_idx[..., None].expand(-1, -1, dim))
    non_pivots = m_raw.gather(-2, non_pivot_idx[..., None].expand(-1, -1, dim))

    k = min(seq - pivot_num, r)
    sim_score, sim_idx = (non_pivots @ pivots.transpose(-2, -1)).max(-1) # (bsz, seq - p)
    dup_idx = sim_score.topk(k, -1, True, False).indices # (bsz, k)
    # sim_idx = pivot_idx.gather(-1, sim_idx).gather(-1, dup_idx)
    dup_idx = non_pivot_idx.gather(-1, dup_idx)

    k = min(seq - r, r)
    imp_mask = torch.zeros_like(indices, dtype=torch.bool).scatter_(-1, imp_idx, True)
    imp_dup_mask = imp_mask.gather(-1, dup_idx)
    # there will be no more than k tokens are both important and duplicated
    # so we select those tokens out here
    imp_dup_mask, imp_dup_idx = imp_dup_mask.to(dtype=torch.int).topk(k, -1, sorted=False) # (bsz, k)
    imp_dup_mask = imp_dup_mask.to(dtype=torch.bool)


    left_idx = indices.scatter(-1, dup_idx, -1).sort(-1).values[:, r:]

    dup_mean_score = imp_score.gather(-1, dup_idx)
    left_mean_score = imp_score.gather(-1, left_idx)
    diff_score = (dup_mean_score.unsqueeze(-1) - left_mean_score.unsqueeze(-2)).abs()



    # diff_score = torch.where(imp_dup_mask, diff_score, torch.inf)
    min_score, min_indices = diff_score.min(dim = -1)

    src_idx = torch.where(imp_dup_mask, dup_idx.gather(-1, imp_dup_idx), -1) + 1
    tar_idx = torch.where(imp_dup_mask, left_idx.gather(-1, min_indices).gather(-1, imp_dup_idx), -1) + 1
    # tar_idx = torch.where(imp_dup_mask, imp_dup_idx, -1) + 1
    # src_idx = torch.where(imp_dup_mask, min_indices.gather(-1, imp_dup_idx), -1) + 1


    # tar_idx = torch.where(imp_dup_value == 1, sim_idx.gather(-1, imp_dup_idx), -1) + 1
    # src_idx = torch.where(imp_dup_value == 1, dup_idx.gather(-1, imp_dup_idx), -1) + 1
    # left_idx = indices.scatter(-1, dup_idx, -1).sort(-1).values[:, r:]

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape
        x_raw_ = torch.cat([torch.zeros_like(x_raw[:, 0:1, :]), x_raw], dim = -2)
        src = x_raw_.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        x_raw_ = x_raw_.scatter_reduce(-2, tar_idx[..., None].expand(-1, -1, dim), src, reduce)
        left = x_raw_[:, 1:].gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)
    return merge(x * size) / size_

    length = x.norm(2, -1, True)
    length_ = merge(length, 'amax')
    x = merge(x)
    x = x / x.norm(2, -1, True) * length_
    return x

# def merge_prune2(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

#     m_prot, m_raw = spliter(metric)

#     # cal improtance by attention score with mean
#     bsz, seq, dim = m_raw.shape

#     imp_num = seq - min(r, seq - r)
#     indices = torch.arange(0, seq, device=metric.device).view(1, seq).expand(bsz, -1)
    
#     mean_token = m_raw.mean(-2, True) # (bsz, 1, dim)
#     mean_norm = mean_token.norm(2, -1, True) # (bsz, 1, 1)
#     mean_token = mean_token / mean_norm

#     metric_norm = m_raw.norm(2, -1, True) # (bsz, seq, 1)
#     m_raw = m_raw / metric_norm
#     imp_score = (m_raw @ mean_token.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
#     imp_idx = imp_score.topk(imp_num, -1, True, False).indices

#     pivot_num = math.ceil(math.log(seq))
#     pivot_idx = metric_norm.squeeze(-1).topk(pivot_num, -1, True, False).indices # (bsz, p)
#     non_pivot_idx = indices[
#         torch.ones_like(indices, dtype=torch.bool).scatter_(-1, pivot_idx, False)
#     ].view(bsz, -1) # (bsz, seq - p)

#     pivots = m_raw.gather(-2, pivot_idx[..., None].expand(-1, -1, dim))
#     non_pivots = m_raw.gather(-2, non_pivot_idx[..., None].expand(-1, -1, dim))

#     k = min(seq - pivot_num, r)
#     sim_score, sim_idx = (non_pivots @ pivots.transpose(-2, -1)).max(-1) # (bsz, seq - p)
#     dup_idx = sim_score.topk(k, -1, True, False).indices # (bsz, k)
#     # sim_idx = pivot_idx.gather(-1, sim_idx).gather(-1, dup_idx)
#     dup_idx = non_pivot_idx.gather(-1, dup_idx)

#     k = min(imp_num, r)
#     imp_mask = torch.zeros_like(indices, dtype=torch.bool).scatter_(-1, imp_idx, True)
#     imp_dup_mask = imp_mask.gather(-1, dup_idx)
#     # there will be no more than k tokens are both important and duplicated
#     # so we select those tokens out here
#     imp_dup_mask, imp_dup_idx = imp_dup_mask.to(dtype=torch.int).topk(k, -1, sorted=False) # (bsz, k)
#     imp_dup_mask = imp_dup_mask.to(dtype=torch.bool)

#     left_idx = indices.scatter(-1, dup_idx, -1).sort(-1).values[:, r:]
#     left_tokens = m_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))

#     src_idx = dup_idx.gather(-1, imp_dup_idx)
#     src_tokens = m_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))

#     score_idx = (src_tokens @ left_tokens.transpose(-2, -1)).max(-1, False).indices
#     tar_idx = left_idx.gather(-1, score_idx)

#     src_idx = torch.where(imp_dup_mask, src_idx, -1) + 1
#     tar_idx = torch.where(imp_dup_mask, tar_idx, -1) + 1

#     def merge(x : torch.Tensor, reduce = 'sum'):
#         x_prot, x_raw = spliter(x)
#         bsz, seq, dim = x_raw.shape
#         x_raw_ = torch.cat([torch.zeros_like(x_raw[:, 0:1, :]), x_raw], dim = -2)
#         src = x_raw_.gather(-2, src_idx[..., None].expand(-1, -1, dim))
#         x_raw_ = x_raw_.scatter_reduce(-2, tar_idx[..., None].expand(-1, -1, dim), src, reduce)
#         left = x_raw_[:, 1:].gather(-2, left_idx[..., None].expand(-1, -1, dim))

#         return torch.cat([x_prot, left], dim = -2)
    
#     handle_source(pinfo, x, merge)
#     size, size_ = handle_size(pinfo, x, merge)
#     length = x.norm(2, -1, True)
#     length_ = merge(length, 'amax')
#     x = merge(x)
#     x = x / x.norm(2, -1, True) * length_
#     return x
#     # return merge(x * size) / size_
#     # return merge(x)

def kidd_pivot(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape

    # the number of important tokens should depend on the number of remaining tokens and reducing tokens.
    # if the number of removing tokens is too large and remaining tokens is too small
    # then most of tokens should be merged rather than pruned, but the number of merging tokens is constrained 
    # by the smaller one of important number and reducing number.
    # Therefore, the number of 

    # num_original = pinfo.get('total_tokens', None)
    # if num_original is None:
    #     pinfo['total_tokens'] = seq
    #     num_original = seq

    num_imp = max(r, seq - r) # min(r, seq // 2)
    num_imp_dup = min(num_imp, r)
    # num_pivot = math.ceil((seq - r) / 20)
    num_pivot = math.ceil(seq / 20)
    # num_non_imp = seq - num_imp
    # num_non_pivot_non_imp = min(seq - num_pivot, num_non_imp)

    if pinfo['class_token']:
        tokens_base = m_prot[:, 0:1]
    else:
        tokens_base = m_raw.mean(-2, True) # (bsz, 1, dim)
    tokens_base = tokens_base / tokens_base.norm(2, -1, True)

    # scale the metric matrix
    metric_norm = m_raw.norm(2, -1, True) # (bsz, seq, 1)
    m_raw = m_raw / metric_norm

    # select pivot tokens, there will be (bsz, num_pivot) indices
    idx_pivot = metric_norm.squeeze(-1).topk(num_pivot, sorted=False).indices
    tokens_pivot = m_raw.gather(-2, idx_pivot[..., None].expand(-1, -1, dim))

    # calculate redundancy
    score_dup, idx_dup_tar = (m_raw @ tokens_pivot.transpose(-2, -1)).max(-1)

    # regard top r tokens as duplicate tokens
    idx_tmp = score_dup.scatter(-1, idx_pivot, -torch.inf).sort(descending=True).indices
    idx_dup = idx_tmp[:, :r]
    left_idx = idx_tmp[:, r:].sort().values

    # now we assign tar_idx and src_idx, note that they are not filtered by importance yet
    tar_idx = idx_pivot.gather(-1, idx_dup_tar).gather(-1, idx_dup)
    src_idx = idx_dup

    # calculate importance now
    score_imp = (m_raw @ tokens_base.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
    idx_imp = score_imp.topk(num_imp, -1, True, False).indices

    mask_imp = torch.zeros_like(score_imp, dtype=torch.bool)
    mask_imp.scatter_(-1, idx_imp, True)
    mask_imp = mask_imp.gather(-1, idx_dup)

    # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
    tar_idx[~mask_imp] = src_idx[~mask_imp]

    # shrunk the array from seq to min(num_imp, r)
    idx_imp_dup = mask_imp.to(dtype=torch.int).topk(num_imp_dup, sorted=False).indices

    tar_idx = tar_idx.gather(-1, idx_imp_dup)
    src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape

        src = x_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        x_raw = x_raw.scatter_reduce(-2, tar_idx[..., None].expand(-1, -1, dim), src, reduce)
        left = x_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    # slerp
    # size, size_ = handle_size(pinfo, x, merge)
    # return merge(x * size) / size_

    # mlerp
    length = x.norm(2, -1, True)
    length_ = merge(length, 'amax')
    x = merge(x)
    x = x / x.norm(2, -1, True)
    x = x * length_
    return x

    # avg
    # size = torch.ones_like(x[..., :1])
    # size_ = merge(size)
    # return merge(x) / merge(size)

    # length_avg
    # length = x.norm(2, -1, True)
    # size, size_ = handle_size(pinfo, x, merge)
    # length_ = merge(length * size) / size_
    # x = merge(x)
    # x = x / x.norm(2, -1, True) * length_
    # return x

def kidd_left(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape

    # the number of important tokens should depend on the number of remaining tokens and reducing tokens.
    # if the number of removing tokens is too large and remaining tokens is too small
    # then most of tokens should be merged rather than pruned, but the number of merging tokens is constrained 
    # by the smaller one of important number and reducing number.
    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil(seq / 20)
    # num_pivot = math.ceil((seq - r) / 20)
    # num_non_imp = seq - num_imp
    # num_non_pivot_non_imp = min(seq - num_pivot, num_non_imp)

    if pinfo['class_token']:
        tokens_base = m_prot[:, 0:1]
    else:
        tokens_base = m_raw.mean(-2, True) # (bsz, 1, dim)
    tokens_base = tokens_base / tokens_base.norm(2, -1, True)

    # scale the metric matrix
    metric_norm = m_raw.norm(2, -1, True) # (bsz, seq, 1)
    m_raw = m_raw / metric_norm

    # select pivot tokens, there will be (bsz, num_pivot) indices
    idx_pivot = metric_norm.squeeze(-1).topk(num_pivot, sorted=False).indices
    tokens_pivot = m_raw.gather(-2, idx_pivot[..., None].expand(-1, -1, dim))

    # calculate redundancy
    score_dup, idx_dup_tar = (m_raw @ tokens_pivot.transpose(-2, -1)).max(-1)
    idx_tmp = score_dup.scatter(-1, idx_pivot, -torch.inf).sort(descending=True).indices

    # regard top r tokens as duplicate tokens
    src_idx = idx_tmp[:, :r]
    left_idx = idx_tmp[:, r:].sort().values

    # find merging target basing on similarity, again, with whole left set
    tokens_src = m_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
    tokens_left = m_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))
    idx_sim = (tokens_left @ tokens_src.transpose(-2, -1)).argmax(-2)
    # assert idx_sim.size(1) == r
    tar_idx = left_idx.gather(-1, idx_sim)

    # calculate importance now
    score_imp = (m_raw @ tokens_base.view(bsz, dim, 1)).squeeze(-1) # (bsz, seq)
    idx_imp = score_imp.topk(num_imp, -1, True, False).indices

    mask_imp = torch.zeros_like(score_imp, dtype=torch.bool)
    mask_imp.scatter_(-1, idx_imp, True)
    mask_imp = mask_imp.gather(-1, src_idx)

    # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
    tar_idx[~mask_imp] = src_idx[~mask_imp]

    # shrunk the array from seq to min(num_imp, r)
    idx_imp_dup = mask_imp.to(dtype=torch.int).topk(num_imp_dup, sorted=False).indices

    tar_idx = tar_idx.gather(-1, idx_imp_dup)
    src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape

        src = x_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        x_raw = x_raw.scatter_reduce(-2, tar_idx[..., None].expand(-1, -1, dim), src, reduce)
        left = x_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    # slerp
    # size, size_ = handle_size(pinfo, x, merge)
    # return merge(x * size) / size_

    # mlerp
    length = x.norm(2, -1, True)
    length_ = merge(length, 'amax')
    x = merge(x)
    x = x / x.norm(2, -1, True)
    x = x * length_
    return x

    # avg
    # size = torch.ones_like(x[..., :1])
    # size_ = merge(size)
    # return merge(x) / merge(size)

    # length_avg
    # length = x.norm(2, -1, True)
    # size, size_ = handle_size(pinfo, x, merge)
    # length_ = merge(length * size) / size_
    # x = merge(x)
    # x = x / x.norm(2, -1, True) * length_
    # return x
