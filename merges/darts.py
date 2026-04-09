import math
import torch
from .utils import clamp, handle_size, handle_source, Spliter

def find_two_metric(variant : str, x, q, k, v):
    left = variant.find('[')
    right = variant.find(']')

    if left < 0 or right < 0 or right <= left:
        return q, k

    sub = variant[left + 1:right]
    subs = [it.strip() for it in sub.split(',')]

    if len(subs) == 0:
        return q, k
    
    table = {'x': x, 'q': q, 'k': k, 'v': v}
    if len(subs) == 1:
        m = table.get(subs[0], q)

        return m, m
    
    metrics = [table.get(it, None) for it in subs]

    return metrics[0], metrics[1]

def apply_dart(pinfo : dict, r : int, variant : str, x : torch.Tensor, q : torch.Tensor, k : torch.Tensor, v):
    prot = 0
    if pinfo.get('class_token', False):
        prot += 1
    if pinfo.get('distill_token', False):
        prot += 1
    
    def split(tensor : torch.Tensor):
        return tensor[..., :prot, :], tensor[..., prot:, :]
    
    seq = x.size(-2)
    seq_ = seq - prot
    r_ = min(r, seq_ - 1)
    if pinfo.get('tome_scheme', False):
        r_ = min(r_, seq_ // 2)

    if r_ <= 0:
        return x
    
    m0, m1 = find_two_metric(variant, x, q.mean(1), k.mean(1), v.mean(1))

    if variant.startswith('dartp'):
        return dart_prune(pinfo, r_, x, m0.detach(), m1.detach(), split)
    else:
        return x

def dart_prune(pinfo, r, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, split : Spliter):

    m0_prot, m0_raw = split(m0)
    m1_prot, m1_raw = split(m1)

    bsz, seq, dim = m1_raw.shape

    pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    k = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)
    # select tokens via m0's l1-norm
    pivot_idx = m0_raw.norm(dim=-1, p=1).topk(k=k, dim=-1).indices
    # but test token's redundancy via m1's cosine similarity
    pivot_tokens = m1_raw.gather(dim=-2, index=pivot_idx[..., None].expand(-1, -1, dim))

    similarity = (pivot_tokens / pivot_tokens.norm(2, -1, True)) @ (m1_raw / m1_raw.norm(2, -1, True)).transpose(-2, -1)
    sim_max_val, sim_max_idx = similarity.max(dim=-2)
    _, src_idx = sim_max_val.scatter(-1, pivot_idx, -1).topk(r, -1)

    tar_idx = sim_max_idx.gather(-1, src_idx)
    tar_idx = pivot_idx.gather(-1, tar_idx)

    left_idx = torch.arange(0, seq, device=x.device)[None, ...]
    left_idx = left_idx.repeat(bsz, 1)
    left_idx = left_idx.scatter(-1, src_idx, 0)
    left_idx = left_idx.sort().values[..., r:]

    def merge(x : torch.Tensor):
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

