import math
import torch
from .utils import Spliter, handle_size, handle_source, clamp, setdiff_indices

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

def apply_cls(pinfo : dict, cinfo : dict, r : int, variant : str, x : torch.Tensor, q : torch.Tensor, k : torch.Tensor, v):
    prot = 0
    if pinfo.get('class_token', False):
        prot += 1
    else:
        print("Warning! Using [CLS] for TC but [CLS] is not existing in this model!")
        return x

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
    
    q_, k_ = find_two_metric(variant, x, q, k, v)

    # (B, H, 1, HD) @ (B, H, DH, N-1) -> (B, H, 1, N-1).mean(1) -> (B, 1, N-1)
    if len(q_.shape) == 4:
        attn_score = (q_[:, :, :1, :] @ k_[:, :, prot:, :].transpose(-2, -1)).mean(1)
    else:
        attn_score = (q_[:, :1, :] @ k_[:, prot:, :].transpose(-2, -1))

    if variant.startswith('clsp'):
        return cls_prune(pinfo, r_, x, attn_score, split)
    elif variant.startswith('clstrpts2'):
        return cls_trpts2(pinfo, r_, x, attn_score, split)
    elif variant.startswith('clstrpts'):
        return cls_trpts(pinfo, r_, x, attn_score, split)
    elif variant.startswith('clsevit'):
        return cls_evit(pinfo, r_, x, q, k, split)
    elif variant.startswith('clsevit2'):
        return cls_evit2(pinfo, r_, x, q, k, split)
    elif variant.startswith('clsm0'):
        return cls_merge0(pinfo, r_, x, attn_score, split)
    elif variant.startswith('clsm1'):
        return cls_merge1(pinfo, r_, x, attn_score, split)
    else:
        return x

def cls_prune(pinfo, r, x, attn, split : Spliter):

    x_prot, x_raw = split(x)

    seq = x_raw.size(1)
    t = seq - r

    top_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)

    top_idx, _ = top_idx.sort(-1, False) # (B, 1, T)
    top_idx = top_idx[..., 0, :, None] # (B, T, 1)

    def merge(it):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        it_raw = torch.gather(it_raw, -2, top_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, it_raw], dim=-2)
    
    handle_size(pinfo, x, merge)
    handle_source(pinfo, x, merge)

    return merge(x)

def cls_merge0(pinfo, r, x, attn, split : Spliter):
    x_prot, x_raw = split(x.detach())

    seq = x_raw.size(1)
    t = seq - r

    left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    src_idx = setdiff_indices(seq, left_idx)

    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    d = x_raw.shape[-1]
    x_tar = x_raw.gather(-2, left_idx.expand(-1, -1, d))
    x_src = x_raw.gather(-2, src_idx.expand(-1, -1, d))

    idx_max = (x_src @ x_tar.transpose(-2, -1)).argmax(-1, True)

    # tar_idx = left_idx.gather(-2, idx_max)
    tar_idx = idx_max

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1)))
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        left = left.scatter_reduce(-2, tar_idx.expand(-1, -1, it.size(-1)), src, mode)

        return torch.cat([it_prot, left], dim=-2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def cls_merge1(pinfo, r, x, attn, split : Spliter):
    x_prot, x_raw = split(x.detach())

    seq = x_raw.size(1)
    t = seq - r

    left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    src_idx = setdiff_indices(seq, left_idx)

    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    # d = x_raw.shape[-1]
    # x_tar = x_raw.gather(-2, left_idx.expand(-1, -1, d))
    # x_src = x_raw.gather(-2, src_idx.expand(-1, -1, d))

    # idx_max = (x_src @ x_tar.transpose(-2, -1)).argmax(-1, True)

    # # tar_idx = left_idx.gather(-2, idx_max)
    # tar_idx = idx_max

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1)))
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, left, src.mean(-2, True)], dim=-2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def cls_trpts(pinfo, r, x, attn, split : Spliter):
    x_prot, x_raw = split(x.detach())

    seq = x_raw.size(1)
    t = seq - r

    left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    src_idx = setdiff_indices(seq, left_idx) # (B, 1, T)

    score = torch.gather(attn, -1, src_idx) # (B, 1, T)

    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    # d = x_raw.shape[-1]
    # x_tar = x_raw.gather(-2, left_idx.expand(-1, -1, d))
    # x_src = x_raw.gather(-2, src_idx.expand(-1, -1, d))

    # idx_max = (x_src @ x_tar.transpose(-2, -1)).argmax(-1, True)

    # # tar_idx = left_idx.gather(-2, idx_max)
    # tar_idx = idx_max

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1))) # (B, T, D)
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, left, (score @ src) / score.sum(-1, True)], dim=-2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def cls_trpts2(pinfo, r, x, attn, split : Spliter):
    x_prot, x_raw = split(x.detach())

    seq = x_raw.size(1)
    r = min(r + 1, seq)
    t = seq - r

    left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    src_idx = setdiff_indices(seq, left_idx) # (B, 1, T)

    score = torch.gather(attn, -1, src_idx) # (B, 1, T)

    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    # d = x_raw.shape[-1]
    # x_tar = x_raw.gather(-2, left_idx.expand(-1, -1, d))
    # x_src = x_raw.gather(-2, src_idx.expand(-1, -1, d))

    # idx_max = (x_src @ x_tar.transpose(-2, -1)).argmax(-1, True)

    # # tar_idx = left_idx.gather(-2, idx_max)
    # tar_idx = idx_max

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1))) # (B, T, D)
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, left, (score @ src) / score.sum(-1, True)], dim=-2)
    
    handle_source(pinfo, x, merge)
    size, size_ = handle_size(pinfo, x, merge)

    return merge(x * size) / size_

def cls_evit(pinfo, r, x, q, k, split : Spliter):
    x_prot, x_raw = split(x.detach())
    q_cls = q[:, :, :1, :]
    _, k_tokens = split(k)

    # (B, H, N, D) @ (B, H, D, 1) -> (B, H, N, 1) -> (B, N, 1) -> (B, 1, N)
    attn = (k_tokens @ q_cls.transpose(-2, -1)).mean(1).transpose(-2, -1) 

    seq = x_raw.size(1)
    r = min(r + 1, seq) # since this method generate append one token to the remaining tokens, we remove one more token for consistant compression ratio as other methods
    t = seq - r

    left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    src_idx = setdiff_indices(seq, left_idx) # (B, 1, T)

    score = torch.gather(attn, -1, src_idx) # (B, 1, T)
    score = torch.softmax(score, dim=-1)

    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1))) # (B, T, D)
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, left, (score @ src)], dim=-2)
    
    handle_source(pinfo, x, merge)
    # size, size_ = handle_size(pinfo, x, merge)

    # return merge(x * size) / size_
    return merge(x)

def cls_evit2(pinfo, r, x, q, k, split : Spliter):
    q_cls = q[:, :, :1, :]
    k_tokens = split(k)

    _, h, seq, hd = k_tokens.shape

    # (B, H, N, D) @ (B, H, D, 1) -> (B, H, N, 1)
    attn = (k @ (q_cls.transpose(-2, -1) / math.sqrt(hd))) #.mean(1).transpose(-2, -1) 
    # (B, H, N, 1) -> (B, N, 1) -> (B, N)
    attn = torch.softmax(attn, -1).unsqueeze(-1).mean(1)
    attn = split(attn)[1]

    # seq = x_raw.size(1)
    r = min(r + 1, seq) # since this method generate append one token to the remaining tokens, we remove one more token for consistant compression ratio as other methods
    t = seq - r

    # left_idx = torch.topk(attn, t, dim=-1, sorted=False).indices # (B, 1, T)
    score_tmp, idx_tmp = torch.sort(attn, -1, True)

    left_idx = idx_tmp[:t]
    src_idx = idx_tmp[t:]
    score = score_tmp[t:]
    
    left_idx, _ = left_idx.sort(-1, False) # (B, 1, T)
    left_idx = left_idx[..., 0, :, None] # (B, T, 1)
    src_idx = src_idx[..., 0, :, None]

    def merge(it, mode = 'sum'):
        it_prot, it_raw = split(it)
        # dim = it.size(-1)
        src = it_raw.gather(-2, src_idx.expand(-1, -1, it.size(-1))) # (B, T, D)
        left = torch.gather(it_raw, -2, left_idx.expand(-1, -1, it.size(-1)))

        return torch.cat([it_prot, left, (score @ src)], dim=-2)
    
    handle_source(pinfo, x, merge)
    # size, size_ = handle_size(pinfo, x, merge)

    # return merge(x * size) / size_
    return merge(x)