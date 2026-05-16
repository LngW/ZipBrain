from typing import Callable, Tuple
import torch
Spliter = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

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

def clamp(x, left, right):
    return max(left, min(x, right))

def select_metric(variant, x, q, k, v):
    left = variant.find('[')
    right = variant.find(']')
    if left > 0 and right > left:
        return {'q': q, 'k': k, 'v': v, 'x': x}.get(variant[left + 1 : right], k)
    else:
        return k

def separate_method_args(m : str):
    left = m.find('[')
    right = m.rfind(']')

    method, args = None, None
    if left > 0 and right > left:
        method = m[:left]
        args = m[left + 1: right]
    else:
        method = m
        args = ''

    args = args.split(',')
    args = [it.strip() for it in args]

    return method, args

def setdiff_indices(total : int, indices : torch.Tensor):
    *b, t0 = indices.shape
    masks = torch.ones((*b, total), dtype=indices.dtype, device=indices.device)
    masks.scatter_(-1, indices, 0)
    masks.cumsum_(-1)
    masks.scatter_(-1, indices, total - 1)

    resorted = masks.scatter(-1, masks - 1, torch.arange(total, device=indices.device).unsqueeze_(0).expand(*b, -1))

    return resorted[..., :total - t0]