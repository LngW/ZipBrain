import os
import math
from typing import Any, NamedTuple
import torch
from torch import Tensor
from einops import rearrange

from .utils import Spliter, handle_source, handle_size, clamp, setdiff_indices

KIDD_RND_DUP=False#(os.getenv('KIDD_RND_DUP', '0') == '1')
KIDD_RND_IMP=False#(os.getenv('KIDD_RND_IMP', '0') == '1')
KIDD_NORM_DUP=True#(os.getenv('KIDD_NORM_DUP', '1') == '1')
KIDD_NORM_MRG=True#(os.getenv('KIDD_NORM_MRG', '1') == '1')
KIDD_NORM_IMP=True#(os.getenv('KIDD_NORM_IMP', '1') == '1')

KIDD_PIVOT_BOTTOM=False#(os.getenv('KIDD_PIVOT_BOTTOM', '0') == '1')

KIDD_MERGE_SCHEME='mlerp'#{'slerp': 'slerp', 'avglen': 'avglen', 'simple':'simple'}.get(os.getenv('KIDD_MERGE_SCHEME', 'mlerp'), 'mlerp')

# basename[arg0,arg1,arg2...]{kwarg0=v0,kwarg1=v1,kwarg2=v2,...}
def separate_method_args(variant : str):

    left0 = variant.find('[')
    right0 = variant.find(']', left0)
    left1 = variant.find('{')
    right1 = variant.find('}', left1)

    method = variant
    args = []
    kwargs = {}

    if left0 > 0 and left1 > 0:
        method = variant[:min(left0, left1)]
    elif left0 > 0:
        method = variant[:left0]
    elif left1 > 0:
        method = method[:left1]

    if left0 > 0 and right0 >= left0:
        # method = variant[:left0]
        args = variant[left0+1:right0].split(',')
        args = [it.strip() for it in args]

    if left1 > 0 and right1 >= left1:
        kws = [it.strip() for it in variant[left1+1:right1].split(',')]

        kwargs = {}
        for kw in kws:
            tmp = kw.split('=')
            if len(tmp) == 1:
                k = tmp[0].strip()
                if k:
                    kwargs[k] = True
            elif len(tmp) == 2:
                k, v = tmp
                k = k.strip()
                v = v.strip()

                if k:
                    kwargs[k] = v
    
    return method, args, kwargs

# In this method, we return three metric, and the three metrics are used for:
# m0: select pivot tokens
# m1: decide the space of judging duplication
# m2: device the space of merging target
# m3: decide the space of judging importance
# We accept three formats: [m0_1_2_3], [m0_1_2, m3], [m0, m1_2, m3] or [m0, m1, m2, m3]
# When not explictly indicated, we use k for m0, m1, m2 and m3
def parse_metric(args : list[str]) -> list[str]:
    if len(args) <= 0:
        return ['k'] * 5
    elif len(args) == 1:
        m0 = args[0]
        return [m0] * 5
    elif len(args) == 2:
        m0_1_2 = args[0]
        m3_4 = args[1]
        return [m0_1_2, m0_1_2, m0_1_2, m3_4, m3_4]
    elif len(args) == 3:
        m0 = args[0]
        m1_2 = args[1]
        m3_4 = args[2]
        return [m0, m1_2, m1_2, m3_4, m3_4]
    elif len(args) == 4:
        m0 = args[0]
        m1 = args[1]
        m2 = args[2]
        m3_4 = args[3]
        return [m0, m1, m2, m3_4, m3_4]
    else:
        return args[:5]

def select_metric(
        args : list[str], 
        x : torch.Tensor, 
        q : torch.Tensor, 
        k : torch.Tensor, 
        v : torch.Tensor
    ) -> list[Tensor]:
    mapping = {
        'x': x.unsqueeze(1), 'q': q.mean(1, True), 'k': k.mean(1, True), 'v': v.mean(1, True),
        'xh': x.unsqueeze(1), 'qh': q, 'kh': k, 'vh': v,

        'xc': x.unsqueeze(1),
        'qc': rearrange(q, 'b (t h) n d -> b t n (h d)', t=1), 
        'kc': rearrange(k, 'b (t h) n d -> b t n (h d)', t=1), 
        'vc': rearrange(v, 'b (t h) n d -> b t n (h d)', t=1),

        'qsh': q.softmax(-1).mean(1, True),
        'qhsh': q.softmax(-1),
        'qcsh': rearrange(q.softmax(-1), 'b (t h) n d -> b t n (h d)', t=1),

        'ksv': k.softmax(-2).mean(1, True),
        'khsv': k.softmax(-2),
        'kcsv': rearrange(k.softmax(-2), 'b (t h) n d -> b t n (h d)', t=1),
    }

    dft = mapping.get('k', k.mean(1, True))

    if len(args) <= 0:
        return [dft] * 5
    elif len(args) == 1:
        m0 = mapping.get(args[0], dft)
        return [m0] * 5
    elif len(args) == 2:
        m0_1_2 = mapping.get(args[0], dft)
        m3_4 = mapping.get(args[1], dft)
        return [m0_1_2, m0_1_2, m0_1_2, m3_4, m3_4]
    elif len(args) == 3:
        m0 = mapping.get(args[0], dft)
        m1_2 = mapping.get(args[1], dft)
        m3_4 = mapping.get(args[2], dft)
        return [m0, m1_2, m1_2, m3_4, m3_4]
    elif len(args) == 4:
        m0 = mapping.get(args[0], dft)
        m1 = mapping.get(args[1], dft)
        m2 = mapping.get(args[2], dft)
        m3_4 = mapping.get(args[2], dft)
        return [m0, m1, m2, m3_4, m3_4]
    else:
        return [mapping.get(it, dft) for it in args[:5]]


def apply_kidd(pinfo : dict, cinfo : dict, r : int, variant : str, x : torch.Tensor, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):

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
    
    method, args, kwargs = separate_method_args(variant)

    if kwargs.get('softmax_qk', False):
        q_ = q.softmax(-1)
        k_ = k.softmax(-2) # why is this used as the kernel function in linear attention?
    else:
        q_ = q
        k_ = k

    # print(args)
    args_ = list(parse_metric(args))

    if kwargs.get('softmax_qk2', False):
        for i in range(3, 5):
            if args_[i] in ['q', 'qh', 'qc']:
                args_[i] = args_[i] + 'sh'
            elif args_[i] in ['k', 'kh', 'kc']:
                args_[i] = args_[i] + 'sv'

    # q, k, v are all (B, N, HD) now, and x is (B, N, D)
    m0, m1, m2, m3, m4 = select_metric(args_, x, q_, k_, v) 

    # if method == 'kidd3mp':
    #     return kidd_pivot3m(pinfo, r_, x, m0, m1, m2, split)
    if method == '':
        return x
    elif method == 'kiddp':
        return kidd_pivot(pinfo, cinfo, r_, x, m0, split)
    elif method == 'kiddl':
        return kidd_left(pinfo, cinfo, r_, x, m0, split)
    elif method == 'kiddl1f':
        cinfo.update(kwargs)
        return kidd_left1f(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split)
    elif method == 'kiddl1a':
        cinfo.update(kwargs)
        assert len(args) == 4, "kiddl1a requires exactly 4 metric args"

        m0 = parse_key_for_a(args[0])
        m1 = parse_key_for_a(args[1])
        m2 = parse_key_for_a(args[2])

        if args_[3].startswith('a'):
            m3_ = args[3]
            m3 = (
                parse_key_for_a(m3_.replace('a', 'q')), 
                parse_key_for_a(m3_.replace('a', 'q'))
                )
        else:
            m3_ = parse_key_for_a(args_[3])
            m3 = (m3_, m3_)

        holder = metric_holder(x.unsqueeze(1), q, k, v)

        return kidd_left1a(pinfo, cinfo, r_, x, seq_, m0, m1, m2, m3, holder, split)
    elif method == 'kiddl2':
        return kidd_left2(pinfo, cinfo, r_, x, m0.mean(1), m1.mean(1), m2.mean(1), m3.mean(1), m4.mean(1), split, False)
    elif method == 'kiddl2f':
        cinfo.update(kwargs)
        return kidd_left2f(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split)
    elif method == 'kiddl3f':
        cinfo.update(kwargs)
        return kidd_left3f(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split)
    # elif method == 'kiddl2pte':
    #     return kidd_left2(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split, True)
    # elif method == 'kiddl2pte':
    #     return kidd_left2(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split, True)
    # elif method == 'kiddl2mh':
        # m0, m1, m2, m3, m4 = select_metric(args, x.unsqueeze(1), q, k, v) # q, k, v are all (B, N, HD) now, and x is (B, N, D)
        # return kidd_left2_mh(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split, False)
    # elif method == 'kiddl2mh_pte':
    #     # m0, m1, m2, m3, m4 = select_metric(args, x.unsqueeze(1), q, k, v) # q, k, v are all (B, H, N, HD) now, and x is (B, 1, N, D)
    #     return kidd_left2_mh(pinfo, cinfo, r_, x, m0, m1, m2, m3, m4, split, True)
    # elif method == 'kiddl4':
    #     return kidd_left4(pinfo, r_, x, m0, split)
    # elif method == 'kiddl3':
    #     return kidd_left3(pinfo, r_, x, m0, m1, m2, m3, split)
    else:
        raise NotImplementedError('Unsupported KIDD variant or args: {}, {}'.format(method, args))

# @torch.compile
def matmul_sum(a : torch.Tensor, b : torch.Tensor):
    return (a @ b.sum(-2).unsqueeze(-1)).squeeze(-1)

# @torch.compile
def matmul_max(a : torch.Tensor, b : torch.Tensor):
    return (a @ b.transpose(-2, -1)).max(-1)

def kidd_pivot(pinfo, cinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)

    with torch.no_grad():
        if use_cls and pinfo['class_token']:
            tokens_base = m_prot[:, 0:1]
        else:
            tokens_base = m_raw.mean(-2, True) # (bsz, 1, dim)
        tokens_base = tokens_base / tokens_base.norm(2, -1, True)

        # scale the metric matrix
        metric_norm = m_raw.norm(2, -1) # (bsz, seq, 1)
        m_raw = m_raw / metric_norm.unsqueeze(-1)

        # select pivot tokens, there will be (bsz, num_pivot) indices
        idx_pivot = metric_norm.topk(num_pivot, sorted=False).indices
        idx_non_pivot = setdiff_indices(seq, idx_pivot)
        tokens_pivot = m_raw.take_along_dim(idx_pivot.unsqueeze(-1), -2)
        tokens_non_pivot = m_raw.take_along_dim(idx_non_pivot.unsqueeze(-1), -2)
        del metric_norm
        # metric_norm = None

        # calculate redundancy
        # score_dup, idx_dup_tar = (m_raw @ tokens_pivot.transpose(-2, -1)).max(-1)
        score_dup, idx_dup_tar = matmul_max(tokens_non_pivot, tokens_pivot)
        # idx_dup_tar = idx_dup_tar.to(dtype=torch.long)

        # score_tmp = torch.empty((bsz, seq), device=score_dup.device, dtype=score_dup.dtype).

        # regard top r tokens as duplicate tokens
        # idx_tmp = score_dup.scatter(-1, idx_pivot, -torch.inf).sort(descending=True).indices
        # idx_tmp = score_dup.sort(descending=True).indices
        idx_tmp = score_dup.topk(r, 1, True, False).indices
        # idx_dup = idx_non_pivot.gather(-1, score_dup.topk(r, -1, True, False).indices)
        # idx_dup = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        # left_idx = setdiff_indices(seq, )

        # now we assign tar_idx and src_idx, note that they are not filtered by importance yet
        tar_idx = idx_pivot.gather(-1, idx_dup_tar.gather(-1, idx_tmp))
        src_idx = idx_non_pivot.gather(-1, idx_tmp)
        left_idx = setdiff_indices(seq, src_idx)

        # calculate importance now
        score_imp = (m_raw @ tokens_base.view(bsz, dim, 1))[:, :, 0] # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        mask_imp = torch.zeros_like(score_imp).scatter(-1, idx_imp, 1.)
        # mask_imp.scatter_(-1, idx_imp, 0)
        mask_imp = mask_imp.gather(-1, src_idx)

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        tar_idx = tar_idx.where(mask_imp > 0, src_idx)

        # shrunk the array from seq to min(num_imp, r)
        # idx_imp_dup = mask_imp.topk(num_imp_dup, sorted=False).indices

        # tar_idx = tar_idx.gather(-1, idx_imp_dup)
        # src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape

        src = x_raw.gather(-2, src_idx.unsqueeze(-1).expand(-1, -1, dim))
        # src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        x_raw = x_raw.scatter_reduce(-2, tar_idx.unsqueeze(-1).expand(-1, -1, dim), src, reduce)
        left = x_raw.gather(-2, left_idx.unsqueeze(-1).expand(-1, -1, dim))
        # left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

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

def kidd_left(pinfo, cinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

    m_prot, m_raw = spliter(metric)

    use_cls = pinfo.get('use_cls', True)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)

    with torch.no_grad():

        if use_cls and pinfo['class_token']:
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
        score_dup, idx_dup_tar = matmul_max(m_raw, tokens_pivot)
        idx_tmp = score_dup.scatter(-1, idx_pivot, -torch.inf).sort(descending=True).indices

        # regard top r tokens as duplicate tokens
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:].sort().values

        # find merging target basing on similarity, again, with whole left set
        tokens_src = m_raw.gather(-2, src_idx[..., None].expand(-1, -1, dim))
        tokens_left = m_raw.gather(-2, left_idx[..., None].expand(-1, -1, dim))
        _, idx_sim = matmul_max(tokens_src, tokens_left) #.argmax(-1)
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

def kidd_left2(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter, pte : bool):

    m0_prot, m0_raw = spliter(m0)
    m1_prot, m1_raw = spliter(m1)
    m2_prot, m2_raw = spliter(m2)
    m3_prot, m3_raw = spliter(m3)
    m4_prot, m4_raw = spliter(m4)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    with torch.no_grad():

        # scale the metric matrix
        metric_norm = m0_raw.norm(2, -1, True) # (bsz, seq, 1)
        if KIDD_PIVOT_BOTTOM:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if KIDD_NORM_DUP:
            dup_space = m1_raw / m1_raw.norm(2, -1, True)
        else:
            dup_space = m1_raw
        tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        if KIDD_RND_DUP:
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) #(dup_space @ tokens_pivot.transpose(-2, -1)).sum(-1)
            # score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        left_idx = idx_tmp[:, r:]

        # src_idx = score_dup.topk(r, sorted=False).indices
        # left_idx = setdiff_indices(seq, src_idx)

        # find merging target basing on similarity, again, with whole left set
        if KIDD_NORM_MRG:
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            merge_space = m2_raw

        tokens_src = merge_space.gather(-2, src_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        tokens_left = merge_space.gather(-2, left_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        if pte:
            pte_space = pinfo['pte']
            pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
            pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
            pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

            _, idx_sim = (tokens_src @ tokens_left.transpose(-2, -1) + pte_src @ pte_left.transpose(-2, -1)).max(-1) #.argmax(-1)
            tar_idx = left_idx.gather(-1, idx_sim)
        else:
            _, idx_sim = matmul_max(tokens_src, tokens_left) #.argmax(-1)
            tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)
        
        if KIDD_NORM_IMP:
            imp_space = m4_raw / m4_raw.norm(2, -1, True)
            tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        else:
            imp_space = m4_raw
        # tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        if KIDD_RND_IMP:
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1) # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        # mask_imp = torch.zeros_like(score_imp, dtype=torch.bool)
        # mask_imp.scatter_(-1, idx_imp, True)
        # mask_imp = mask_imp.gather(-1, src_idx)
        # mask_imp_1 = torch.isin(src_idx, idx_imp)
        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)

        # print(torch.allclose(mask_imp, mask_imp_2))

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        tar_idx = tar_idx.where(mask_imp, src_idx)

        # shrunk the array from seq to min(num_imp, r)
        # idx_imp_dup = mask_imp.to(dtype=torch.int).topk(num_imp_dup, sorted=False).indices

        # tar_idx = tar_idx.gather(-1, idx_imp_dup)
        # src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x
        bsz, seq, dim = x_raw.shape

        if True:
            src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
            tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
            x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
            left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)
        else:
            x_raw = x_raw.transpose(-2, -1).contiguous() # now it is bsz, dim, seq
            src = x_raw.take_along_dim(src_idx.unsqueeze(-2), -1) # * mask_imp.unsqueeze(-2)

            # Version 0
            if True: #reduce != 'sum':
                tar_idx_ = tar_idx.unsqueeze(-2).expand(-1, dim, -1)
                x_raw = x_raw.scatter_reduce(-1, tar_idx_, src, reduce)
            # Version 1
            else:
                batch_offsets = torch.arange(bsz, device=tar_idx.device, dtype=torch.long).view(bsz, 1, 1) * (dim * seq)
                dim_offsets = torch.arange(dim, device=tar_idx.device, dtype=torch.long).view(1, dim, 1) * seq
                flat_idx = (tar_idx.unsqueeze(-2) + batch_offsets + dim_offsets).flatten()
                flat_src = src.flatten()
                flat_x_raw = x_raw.flatten()
                flat_x_raw = flat_x_raw.index_add(0, flat_idx, flat_src)
                x_raw = flat_x_raw.view(bsz, dim, seq)

            left = x_raw.take_along_dim(left_idx.unsqueeze(-2), -1)
            left = left.transpose(-2, -1)#.contiguous()

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split=False)

    # slerp
    if KIDD_MERGE_SCHEME == 'slerp':
        size, size_ = handle_size(pinfo, x, merge)
        return merge(x * size) / size_

    # mlerp
    elif KIDD_MERGE_SCHEME == 'mlerp':
        length = x.norm(2, -1, True)
        length_ = merge(length, 'amax')
        x = merge(x)
        x = x / x.norm(2, -1, True)
        x = x * length_
        return x

    # avg
    elif KIDD_MERGE_SCHEME == 'simple':
        size = torch.ones_like(x[..., :1])
        size_ = merge(size)
        return merge(x) / merge(size)

    # length_avg
    elif KIDD_MERGE_SCHEME == 'avglen':
        length = x.norm(2, -1, True)
        size, size_ = handle_size(pinfo, x, merge)
        length_ = merge(length * size) / size_
        x = merge(x)
        x = x / x.norm(2, -1, True) * length_
        return x

def kidd_left2f(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter):

    # note that x is of shape (B, N, D), but m0~m4 are of shape (B, H, N, HD), and H and HD may vary

    m0_prot, m0_raw = spliter(m0) # for pivot selection
    m1_prot, m1_raw = spliter(m1) # to form the duplication space
    m2_prot, m2_raw = spliter(m2) # to calc pairs of merging.
    m3_prot, m3_raw = spliter(m3) # to form the base token for importance. [CLS] or mean of tokens
    m4_prot, m4_raw = spliter(m4) # the other part of importance calculation

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, _, seq, _ = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    pte = cinfo.get('pte', False)
    adjust = cinfo.get('adjust', False)

    with torch.no_grad():

        # scale the metric matrix
        norm_p = int(cinfo.get('m0p', 2))
        metric_norm = m0_raw.norm(norm_p, -1, True).mean(1) # (bsz, head, seq, 1) -> (bsz, seq, 1)
        if cinfo.get('bottom_pivot', KIDD_PIVOT_BOTTOM):
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # idx_pivot is of shape (bsz, num_pivot)

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if cinfo.get('norm_dup', KIDD_NORM_DUP): # The normalization is quite necessary here
            dup_space = m1_raw / m1_raw.norm(2, -1, True) # (bsz, head, seq, dim)
        else:
            dup_space = m1_raw
        # tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        tokens_pivot = dup_space.take_along_dim(idx_pivot.view(bsz, 1, num_pivot, 1), -2)
        if cinfo.get('rnd_dup', KIDD_RND_DUP):
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) # (b, h, n, d) @ (b h d num_pivot) -> (b h n num_pivot) -> (b h n)
            score_dup = score_dup.mean(1) # (b, h, n) -> (b n)

            # adjust:
            if adjust:
                mask_pivot = torch.zeros((bsz, seq), device=score_dup.device, dtype=torch.int).scatter(-1, idx_pivot, 1)
                score_dup = score_dup - mask_pivot
                score_dup = score_dup * (mask_pivot / seq + 1)
            
        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        left_idx = idx_tmp[:, r:]

        if cinfo.get('norm_merge', KIDD_NORM_MRG):
            # find merging target basing on similarity
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            # Without normalization, we are calculating projections?
            merge_space = m2_raw

        tokens_src = merge_space.take_along_dim(src_idx.reshape(bsz, 1, r, 1), -2) # (bsz, head, r, dim)
        tokens_left = merge_space.take_along_dim(left_idx.reshape(bsz, 1, seq - r, 1), dim=-2) # (bsz, head, left, dim)

        score_tgt = (tokens_left @ tokens_src.transpose(-2, -1)).mean(1)

        if pte:
            pte_space = pinfo['pte']
            # pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
            pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
            pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

            score_tgt = score_tgt + pte_left @ pte_src.transpose(-2, -1)

        idx_sim = score_tgt.argmax(-2)
        tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)

        # It is not necessary to normalize the tokens_base since they share the same ||tokens_base||        
        
        if cinfo.get('norm_imp', KIDD_NORM_IMP) not in [False, 'false', 'False', '0']:
            imp_space = m4_raw / m4_raw.norm(2, -1, True)
        else:
            imp_space = m4_raw

        if cinfo.get('rnd_imp', KIDD_RND_IMP):
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1).mean(1) # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)
        tar_idx = tar_idx.where(mask_imp, src_idx)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x

        bsz, seq, dim = x_raw.shape
        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split=False)

    merge_scheme = cinfo.get('merge_scheme', KIDD_MERGE_SCHEME)
    merge_scheme = {'slerp': 'slerp', 'avglen': 'avglen', 'simple':'simple'}.get(merge_scheme, 'mlerp')

    # slerp
    if merge_scheme == 'slerp':
        size, size_ = handle_size(pinfo, x, merge)
        return merge(x * size) / size_

    # mlerp
    elif merge_scheme == 'mlerp':
        length = x.norm(2, -1, True)
        length_ = merge(length, 'amax')
        x = merge(x)
        x = x / x.norm(2, -1, True)
        x = x * length_
        return x

    # avg
    elif merge_scheme == 'simple':
        size = torch.ones_like(x[..., :1])
        size_ = merge(size)
        return merge(x) / merge(size)

    # length_avg
    elif merge_scheme == 'avglen':
        length = x.norm(2, -1, True)
        size, size_ = handle_size(pinfo, x, merge)
        length_ = merge(length * size) / size_
        x = merge(x)
        x = x / x.norm(2, -1, True) * length_
        return x

def kidd_left1f(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter):
    # A copy of kidd_left2f, but the pivot selection process is the version of kidd_left1

    # note that x is of shape (B, N, D), but m0~m4 are of shape (B, H, N, HD), and H and HD may vary

    m0_prot, m0_raw = spliter(m0) # for pivot selection
    m1_prot, m1_raw = spliter(m1) # to form the duplication space
    m2_prot, m2_raw = spliter(m2) # to calc pairs of merging.
    m3_prot, m3_raw = spliter(m3) # to form the base token for importance. [CLS] or mean of tokens
    m4_prot, m4_raw = spliter(m4) # the other part of importance calculation

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    bsz, _, seq, _ = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    # define the default number of important tokens and pivot tokens, and the duplication factor
    num_imp = max(r, seq - r)
    num_pivot = math.ceil((seq - r) * 0.05)
    # num_imp_dup = min(num_imp, r)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)

    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)

    pte = cinfo.get('pte', False)
    adjust = cinfo.get('adjust', False)
    mh_amax = cinfo.get('mh_amax', False)
    mh_amax_pivot = mh_amax or cinfo.get('mh_amax_pivot', False)

    with torch.no_grad():

        if cinfo.get('rnd_dup', KIDD_RND_DUP):
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            # scale the metric matrix
            norm_p = int(cinfo.get('m0p', 2))
            metric_norm = m0_raw.norm(norm_p, -1, True) #.mean(1) # (bsz, head, seq, 1) -> (bsz, seq, 1)
            metric_norm = metric_norm.amax(1) if mh_amax_pivot else metric_norm.mean(1)
            if cinfo.get('bottom_pivot', KIDD_PIVOT_BOTTOM):
                idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
            else:
                idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

            # now idx_pivot is of shape (bsz, num_pivot)

            # calculate redundancy
            if cinfo.get('norm_dup', KIDD_NORM_DUP): # The normalization is quite necessary here
                dup_space = m1_raw / m1_raw.norm(2, -1, True) # (bsz, head, seq, dim)
            else:
                dup_space = m1_raw

            # select pivot tokens, there will be (bsz, num_pivot) indices
            tokens_pivot = dup_space.take_along_dim(idx_pivot.view(bsz, 1, num_pivot, 1), -2)
            # (b, h, n, d) @ (b h d num_pivot) -> (b h n num_pivot) -> (b h n)
            score_dup = matmul_sum(dup_space, tokens_pivot) 
            if mh_amax:
                score_dup = score_dup.amax(1) # (b, h, n) -> (b n)
            else:
                score_dup = score_dup.mean(1) # (b, h, n) -> (b n)

            # pivots can not be treated as redundant tokens
            score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)
        

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:]
        
        if cinfo.get('rnd_merge', False):
            score_tgt = torch.rand(bsz, seq-r, r, device=m2_raw.device)
        else:
            if cinfo.get('norm_merge', KIDD_NORM_MRG):
                # find merging target basing on similarity
                merge_space = m2_raw / m2_raw.norm(2, -1, True)
            else:
                # Without normalization, we are calculating projections?
                merge_space = m2_raw

            tokens_src = merge_space.take_along_dim(src_idx.reshape(bsz, 1, r, 1), -2) # (bsz, head, r, dim)
            tokens_left = merge_space.take_along_dim(left_idx.reshape(bsz, 1, seq - r, 1), dim=-2) # (bsz, head, left, dim)

            score_tgt = (tokens_left @ tokens_src.transpose(-2, -1))
            
            if mh_amax:
                score_tgt = score_tgt.amax(1)
            else:
                score_tgt = score_tgt.mean(1)

            if pte:
                pte_space = pinfo['pte']
                # pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
                pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
                pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

                score_tgt = score_tgt + pte_left @ pte_src.transpose(-2, -1)

        idx_sim = score_tgt.argmax(-2)
        tar_idx = left_idx.gather(-1, idx_sim)

        if num_imp < seq:
            if cinfo.get('rnd_imp', KIDD_RND_IMP):
                score_imp = torch.rand(bsz, seq, device=x.device)
            else:
                # calculate importance now
                if use_cls and pinfo['class_token']:
                    tokens_base = m3_prot[:, 0:1]
                else:
                    tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)

                # It is not necessary to normalize the tokens_base since they share the same ||tokens_base||        
                # So that we can save some computation.
                if cinfo.get('norm_imp', KIDD_NORM_IMP) not in [False, 'false', 'False', '0']:
                    imp_space = m4_raw / m4_raw.norm(2, -1, True)
                else:
                    imp_space = m4_raw

                score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1)
                score_imp = score_imp.amax(1) if mh_amax else score_imp.mean(1) # (bsz, seq)
            idx_imp = score_imp.topk(num_imp, -1, True, False).indices

            # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
            mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)
            # Also pivot tokens should not be discarded, so we need to make sure that they are important
            # mask_imp = mask_imp | (src_idx.unsqueeze(-1) == idx_pivot.unsqueeze(-2)).any(-1)
            tar_idx = tar_idx.where(mask_imp, src_idx)
        else:
            # otherwise, all tokens are important, so all of them need to be merged
            pass

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x

        bsz, seq, dim = x_raw.shape
        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split=False)

    merge_scheme = cinfo.get('merge_scheme', KIDD_MERGE_SCHEME)
    merge_scheme = {'slerp': 'slerp', 'avglen': 'avglen', 'simple':'simple'}.get(merge_scheme, 'mlerp')

    # slerp
    if merge_scheme == 'slerp':
        size, size_ = handle_size(pinfo, x, merge)
        return merge(x * size) / size_

    # mlerp
    elif merge_scheme == 'mlerp':
        length = x.norm(2, -1, True)
        length_ = merge(length, 'amax')
        x = merge(x)
        x = x / x.norm(2, -1, True)
        x = x * length_
        return x

    # avg
    elif merge_scheme == 'simple':
        size = torch.ones_like(x[..., :1])
        size_ = merge(size)
        return merge(x) / merge(size)

    # length_avg
    elif merge_scheme == 'avglen':
        length = x.norm(2, -1, True)
        size, size_ = handle_size(pinfo, x, merge)
        length_ = merge(length * size) / size_
        x = merge(x)
        x = x / x.norm(2, -1, True) * length_
        return x

class metric_config(NamedTuple):
    base : str
    suffix : str
    max_min_instead_of_mean : bool = False

def parse_key_for_a(name : str) -> metric_config:
    name = name.strip()

    length = len(name)

    if length == 0:
        raise Exception("Can not parse an empty metric name")
    
    elif length == 1:
        if name in {'x','q','k','v'}:
            return metric_config(name, '', False)
        else:
            raise Exception(f"Unexpected name {name}")
        
    elif length == 2:
        base, suffix = name

        assert base in {'x', 'q', 'k', 'v'}, f"Unexpected name {name}"
        assert suffix in {'', 'c', 'h', 'm'}, f"Unexpected name {name}"

        min_max = suffix == 'm'

        if base == 'x':
            return metric_config('x', '', min_max)
        elif suffix in {'h', 'm'}:
            return metric_config(base, 'h', min_max)
        elif suffix == 'c':
            return metric_config(base, 'c', False)
        
    raise Exception(f"Unexpected length or suffix in processing metric name: {name}")
    

class metric_holder:

    def __init__(self, x : Tensor, q : Tensor, k : Tensor, v : Tensor):
        self.cache = {}
        self.x = x
        self.q = q
        self.k = k
        self.v = v

    def retrive(self, cfg : metric_config) -> Tensor:
        key = "{}{}".format(cfg.base, cfg.suffix)
        if key in self.cache:
            return self.cache[key]
        
        value = {'x':self.x, 'q':self.q, 'k':self.k, 'v':self.v}.get(cfg.base, None)
        assert value is not None

        if cfg.suffix == '':
            value = value.mean(1, True)
        elif cfg.suffix == 'h':
            value = value
        elif cfg.suffix == 'c':
            value = value.permute(0, 2, 1, 3).flatten(2).unsqueeze(1)

        self.cache[key] = value
        return value

def kidd_left1a(
        pinfo : dict[str, Any], 
        cinfo : dict[str, Any], 
        r : int, 
        x : Tensor, 
        seq_ : int,
        m0 : metric_config, 
        m1 : metric_config, 
        m2 : metric_config, 
        m3 : tuple[metric_config, metric_config], 
        holder : metric_holder,
        spliter : Spliter
    ):
    # A copy of kidd_left1f, but now we support only 4 metric parameters. 
    # m0, m1 and m2 are all in ['x'] + [it + suf for it in 'qkv' for suf in ['', 'h', 'm']]
    # m3 is in ['x'] + [it + suf for it in 'aqkv' for suf in ['', 'h', 'm']], here 'a' is to calc the attention score, and qkv will calc cosine similarity

    # note that x is of shape (B, N, D), but m0~m4 are of shape (B, H, N, HD), and H and HD may vary

    use_cls = pinfo.get('use_cls', False)

    # here the seq is the length of non protected tokens.
    bsz, seq = x.size(0), seq_

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    # define the default number of important tokens and pivot tokens, and the duplication factor
    num_imp = max(r, seq - r)
    num_pivot = math.ceil((seq - r) * 0.05)
    # num_imp_dup = min(num_imp, r)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)

    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)

    # other possible components
    pte = cinfo.get('pte', False)

    with torch.no_grad():
        # 1, calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if cinfo.get('rnd_dup', KIDD_RND_DUP):
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            # 1. generate indices for pivot token selection, basing on the norm of the metric matrix.
            m0_prot, m0_raw = spliter(holder.retrive(m0)) # for pivot selection
            # scale the metric matrix
            norm_p = int(cinfo.get('m0p', 2))
            metric_norm = m0_raw.norm(norm_p, -1, True) # (bsz, head, seq, 1)
            if m0.max_min_instead_of_mean:
                metric_norm = metric_norm.amax(1) # (bsz, seq, 1)
            else:
                metric_norm = metric_norm.mean(1) # (bsz, seq, 1)

            if cinfo.get('bottom_pivot', KIDD_PIVOT_BOTTOM):
                idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
            else:
                idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices
            
            # 2. define on which space we will calculate the redundancy score.
            m1_prot, m1_raw = spliter(holder.retrive(m1)) # to form the duplication space
            if cinfo.get('norm_dup', KIDD_NORM_DUP): # The normalization is quite necessary here, so normally we should not disable it.
                dup_space = m1_raw / m1_raw.norm(2, -1, True) # (bsz, head, seq, dim)
            else:
                dup_space = m1_raw

            # 3. calculate the redundancy score
            tokens_pivot = dup_space.take_along_dim(idx_pivot.view(bsz, 1, num_pivot, 1), -2)
            # (b, h, n, d) @ (b, h, d, num_pivot) -> (b, h, n, num_pivot) -> (b h n)
            score_dup = matmul_sum(dup_space, tokens_pivot) 
            if m1.max_min_instead_of_mean:
                score_dup = score_dup.amax(1) # (b, h, n) -> (b n)
            else:
                score_dup = score_dup.mean(1) # (b, h, n) -> (b n)
            score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)

        # 2. We define which tokens to be reduced basing on the score_dup.
        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:]

        # 3. Find merge target for each token in src_idx
        m2_prot, m2_raw = spliter(holder.retrive(m2)) # to calc pairs of merging.
        if cinfo.get('norm_merge', KIDD_NORM_MRG):
            # find merging target basing on similarity
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            # Without normalization, we are calculating projections?
            merge_space = m2_raw

        tokens_src = merge_space.take_along_dim(src_idx.reshape(bsz, 1, r, 1), -2) # (bsz, head, r, dim)
        tokens_left = merge_space.take_along_dim(left_idx.reshape(bsz, 1, seq - r, 1), dim=-2) # (bsz, head, left, dim)

        score_tgt = (tokens_left @ tokens_src.transpose(-2, -1)) #.mean(1)
        if m2.max_min_instead_of_mean:
            score_tgt = score_tgt.amax(1)
        else:
            score_tgt = score_tgt.mean(1)

        if pte:
            pte_space = pinfo['pte']
            # pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
            pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
            pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

            score_tgt = score_tgt + pte_left @ pte_src.transpose(-2, -1)

        idx_sim = score_tgt.argmax(-2)
        tar_idx = left_idx.gather(-1, idx_sim)

        # 3. Judge the importance
        if cinfo.get('rnd_imp', KIDD_RND_IMP):
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            m30, m31 = m3
            imp_base = holder.retrive(m30)
            if use_cls:
                imp_base = spliter(imp_base)[0][0:1]
            else:
                imp_base = spliter(imp_base)[1].mean(-2, True)

            imp_space = holder.retrive(m31)
            norm_imp = cinfo.get('norm_imp', KIDD_NORM_IMP) not in [False, 'false', 'False', '0']
            if norm_imp:
                imp_space = imp_space / imp_space.norm(2, -1, True)

            # (b,h,n,d) @ (b,h,d,1) -> (b,h,n,1) -> (b,h,n)
            score_imp = (imp_space @ imp_base.transpose(-2, -1)).squeeze(-1)
            
            # (b, h, seq) -> (b, seq)
            if m31.max_min_instead_of_mean:
                score_imp = score_imp.amax(1)
            else:
                score_imp = score_imp.mean(1)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices
        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # pivotal tokens should be also treated as important tokens, but pivotal tokens can not 
        # be selected in src_idx, so we can freely assume that they are safe.
        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)

        # apply mask_imp to tar_idx, now, when an token is important, then it will have a target in left_idx, or its target will be itself
        # After reduction, tokens in src_idx will be removed, therefore, only important tokens are merged, and others are discarded.
        tar_idx = tar_idx.where(mask_imp, src_idx)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x

        bsz, seq, dim = x_raw.shape
        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split=False)

    merge_scheme = cinfo.get('merge_scheme', KIDD_MERGE_SCHEME)
    merge_scheme = {'slerp': 'slerp', 'avglen': 'avglen', 'simple':'simple'}.get(merge_scheme, 'mlerp')

    # slerp
    if merge_scheme == 'slerp':
        size, size_ = handle_size(pinfo, x, merge)
        return merge(x * size) / size_

    # mlerp
    elif merge_scheme == 'mlerp':
        length = x.norm(2, -1, True)
        length_ = merge(length, 'amax')
        x = merge(x)
        x = x / x.norm(2, -1, True)
        x = x * length_
        return x

    # avg
    elif merge_scheme == 'simple':
        size = torch.ones_like(x[..., :1])
        size_ = merge(size)
        return merge(x) / merge(size)

    # length_avg
    elif merge_scheme == 'avglen':
        length = x.norm(2, -1, True)
        size, size_ = handle_size(pinfo, x, merge)
        length_ = merge(length * size) / size_
        x = merge(x)
        x = x / x.norm(2, -1, True) * length_
        return x

# Inverted version of kiddl2f, discard bottom r important tokens
def kidd_left3f(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter):

    # note that x is of shape (B, N, D), but m0~m4 are of shape (B, H, N, HD), and H and HD may vary

    m0_prot, m0_raw = spliter(m0) # for pivot selection
    m1_prot, m1_raw = spliter(m1) # to form the duplication space
    m2_prot, m2_raw = spliter(m2) # to calc pairs of merging.
    m3_prot, m3_raw = spliter(m3) # to form the base token for importance. [CLS] or mean of tokens
    m4_prot, m4_raw = spliter(m4) # the other part of importance calculation

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, _, seq, _ = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    pte = cinfo.get('pte', False)
    adjust = cinfo.get('adjust', False)

    with torch.no_grad():

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)

        # It is not necessary to normalize the tokens_base since they share the same ||tokens_base||        
        
        if cinfo.get('norm_imp', KIDD_NORM_IMP) not in [False, 'false', 'False', '0']:
            imp_space = m4_raw / m4_raw.norm(2, -1, True)
        else:
            imp_space = m4_raw

        if cinfo.get('rnd_imp', KIDD_RND_IMP):
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1).mean(1) # (bsz, seq)

        idx_tmp = score_imp.argsort(-1, True)
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:]
        # left_idx = idx_tmp[:, r:].sort().values

        if cinfo.get('norm_merge', KIDD_NORM_MRG):
            # find merging target basing on similarity
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            # Without normalization, we are calculating projections?
            merge_space = m2_raw

        tokens_src = merge_space.take_along_dim(src_idx.reshape(bsz, 1, r, 1), -2) # (bsz, head, r, dim)
        tokens_left = merge_space.take_along_dim(left_idx.reshape(bsz, 1, seq - r, 1), dim=-2) # (bsz, head, left, dim)

        score_tgt = (tokens_left @ tokens_src.transpose(-2, -1)).mean(1)

        if pte:
            pte_space = pinfo['pte']
            # pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
            pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
            pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

            score_tgt = score_tgt + pte_left @ pte_src.transpose(-2, -1)

        idx_sim = score_tgt.argmax(-2)
        tar_idx = left_idx.gather(-1, idx_sim)

        # scale the metric matrix
        norm_p = int(cinfo.get('m0p', 2))
        metric_norm = m0_raw.norm(norm_p, -1, True).mean(1) # (bsz, head, seq, 1) -> (bsz, seq, 1)
        if cinfo.get('bottom_pivot', KIDD_PIVOT_BOTTOM):
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # idx_pivot is of shape (bsz, num_pivot)

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if cinfo.get('norm_dup', KIDD_NORM_DUP): # The normalization is quite necessary here
            dup_space = m1_raw / m1_raw.norm(2, -1, True) # (bsz, head, seq, dim)
        else:
            dup_space = m1_raw
        # tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        tokens_pivot = dup_space.take_along_dim(idx_pivot.view(bsz, 1, num_pivot, 1), -2)
        if cinfo.get('rnd_dup', KIDD_RND_DUP):
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) # (b, h, n, d) @ (b h d num_pivot) -> (b h n num_pivot) -> (b h n)
            score_dup = score_dup.mean(1) # (b, h, n) -> (b n)

            # adjust:
            if adjust:
                mask_pivot = torch.zeros((bsz, seq), device=score_dup.device, dtype=torch.int).scatter(-1, idx_pivot, 1)
                score_dup = score_dup - mask_pivot
                score_dup = score_dup * (mask_pivot / seq + 1)
            
        # calculate importance now
        idx_dup = score_dup.topk(num_imp, -1, True, False).indices

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        mask_imp = (src_idx.unsqueeze(-1) == idx_dup.unsqueeze(-2)).any(-1)
        tar_idx = tar_idx.where(mask_imp, src_idx)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x

        bsz, seq, dim = x_raw.shape
        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split=False)

    merge_scheme = cinfo.get('merge_scheme', KIDD_MERGE_SCHEME)
    merge_scheme = {'slerp': 'slerp', 'avglen': 'avglen', 'simple':'simple'}.get(merge_scheme, 'mlerp')

    # slerp
    if merge_scheme == 'slerp':
        size, size_ = handle_size(pinfo, x, merge)
        return merge(x * size) / size_

    # mlerp
    elif merge_scheme == 'mlerp':
        length = x.norm(2, -1, True)
        length_ = merge(length, 'amax')
        x = merge(x)
        x = x / x.norm(2, -1, True)
        x = x * length_
        return x

    # avg
    elif merge_scheme == 'simple':
        size = torch.ones_like(x[..., :1])
        size_ = merge(size)
        return merge(x) / merge(size)

    # length_avg
    elif merge_scheme == 'avglen':
        length = x.norm(2, -1, True)
        size, size_ = handle_size(pinfo, x, merge)
        length_ = merge(length * size) / size_
        x = merge(x)
        x = x / x.norm(2, -1, True) * length_
        return x

def kidd_left2_mh(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter, pte : bool):

    m0_prot, m0_raw = spliter(m0)
    m1_prot, m1_raw = spliter(m1)
    m2_prot, m2_raw = spliter(m2)
    m3_prot, m3_raw = spliter(m3)
    m4_prot, m4_raw = spliter(m4)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, _, seq, _ = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    with torch.no_grad():

        # scale the metric matrix
        metric_norm = m0_raw.norm(1, -1, True).mean(1) # (bsz, seq, 1)
        if KIDD_PIVOT_BOTTOM:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        m1_raw = m1_raw.permute(0, 2, 1, 3)
        m1_raw = m1_raw.reshape(m1_raw.size(0), m1_raw.size(1), -1)
        if KIDD_NORM_DUP:
            dup_space = m1_raw / m1_raw.norm(2, -1, True)
        else:
            dup_space = m1_raw
        tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        if KIDD_RND_DUP:
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) #(dup_space @ tokens_pivot.transpose(-2, -1)).sum(-1)
            # score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        left_idx = idx_tmp[:, r:]

        # src_idx = score_dup.topk(r, sorted=False).indices
        # left_idx = setdiff_indices(seq, src_idx)

        # find merging target basing on similarity, again, with whole left set
        m2_raw = m2_raw.mean(1)
        if KIDD_NORM_MRG:
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            merge_space = m2_raw
        tokens_src = merge_space.gather(-2, src_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        tokens_left = merge_space.gather(-2, left_idx[..., None].expand(-1, -1, merge_space.size(-1)))

        if pte:

            pte_space = pinfo['pte']
            pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
            pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
            pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

            _, idx_sim = (tokens_src @ tokens_left.transpose(-2, -1) + pte_src @ pte_left.transpose(-2, -1)).max(-1) #.argmax(-1)
            tar_idx = left_idx.gather(-1, idx_sim)

        else:

            _, idx_sim = matmul_max(tokens_src, tokens_left) #.argmax(-1)
            tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, :, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, h, 1, dim)
        
        if KIDD_NORM_IMP:
            imp_space = m4_raw / m4_raw.norm(2, -1, True)
            tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        else:
            imp_space = m4_raw
        # tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        if KIDD_RND_IMP:
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).mean(1).squeeze(-1) # (bsz, seq)
            score_imp = score_imp
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices
        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)
        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        tar_idx = tar_idx.where(mask_imp, src_idx)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[..., 0:0, :], x
        bsz, seq, dim = x_raw.shape

        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)
    if pte:
        pinfo['pte'] = merge(pinfo['pte'], do_split = False)

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

def kidd_left3(pinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, spliter : Spliter):

    m0_prot, m0_raw = spliter(m0)
    m1_prot, m1_raw = spliter(m1)
    m2_prot, m2_raw = spliter(m2)
    m3_prot, m3_raw = spliter(m3)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = r // 2
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in pinfo:
        imp_num = pinfo['imp_num'].pop(0)
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in pinfo:
        imp_factor = pinfo['imp_factor'].pop(0)
        if imp_factor is not None:
            num_imp = clamp(math.floor((r + 1) * imp_factor), 0, r)
    if 'pivot_num' in pinfo:
        pivot_num = pinfo['pivot_num'].pop(0)
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in pinfo:
        pivot_factor = pinfo['pivot_factor'].pop(0)
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    with torch.no_grad():

        # scale the metric matrix
        metric_norm = m0_raw.norm(2, -1, True) # (bsz, seq, 1)
        if KIDD_PIVOT_BOTTOM:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if KIDD_NORM_DUP:
            dup_space = m1_raw / m1_raw.norm(2, -1, True)
        else:
            dup_space = m1_raw
        tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        if KIDD_RND_DUP:
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) #(dup_space @ tokens_pivot.transpose(-2, -1)).sum(-1)
            # score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:].sort().values

        # src_idx = score_dup.topk(r, sorted=False).indices
        # left_idx = setdiff_indices(seq, src_idx)

        # find merging target basing on similarity, again, with whole left set
        if KIDD_NORM_MRG:
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            merge_space = m2_raw

        tokens_src = merge_space.gather(-2, src_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        tokens_left = merge_space.gather(-2, left_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        _, idx_sim = matmul_max(tokens_src, tokens_left) #.argmax(-1)
        # assert idx_sim.size(1) == r
        tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)
        
        if KIDD_NORM_IMP:
            imp_space = m3_raw / m3_raw.norm(2, -1, True)
        else:
            imp_space = m3_raw
        tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        if KIDD_RND_IMP:
            score_imp = torch.rand(bsz, r, device=x.device)
        else:
            # score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1) # (bsz, seq)
            score_imp = (imp_space.take_along_dim(src_idx.unsqueeze(-1), -2) @ tokens_base.transpose(-2, -1)).squeeze(-1)
        # idx_imp = score_imp.topk(num_imp, -1, True, False).indices
        idx_non_imp = score_imp.topk(num_imp, -1, False, False).indices
        idx_non_imp = src_idx.gather(-1, idx_non_imp)

        mask_imp = torch.ones_like(imp_space[..., 0], dtype=torch.bool)
        mask_imp.scatter_(-1, idx_non_imp, False)
        mask_imp = mask_imp.gather(-1, src_idx)

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        tar_idx = tar_idx.where(mask_imp, src_idx)

        # shrunk the array from seq to min(num_imp, r)
        # idx_imp_dup = mask_imp.to(dtype=torch.int).topk(num_imp_dup, sorted=False).indices

        # tar_idx = tar_idx.gather(-1, idx_imp_dup)
        # src_idx = src_idx.gather(-1, idx_imp_dup)

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

def kidd_left4(pinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, spliter : Spliter):

    m0_prot, m0_raw = spliter(m0)
    # m1_prot, m1_raw = spliter(m1)
    # m2_prot, m2_raw = spliter(m2)
    # m3_prot, m3_raw = spliter(m3)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in pinfo:
        imp_num = pinfo['imp_num'].pop(0)
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in pinfo:
        imp_factor = pinfo['imp_factor'].pop(0)
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in pinfo:
        pivot_num = pinfo['pivot_num'].pop(0)
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in pinfo:
        pivot_factor = pinfo['pivot_factor'].pop(0)
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    with torch.no_grad():

        # scale the metric matrix
        metric_norm = m0_raw.norm(2, -1, True) # (bsz, seq, 1)
        if KIDD_PIVOT_BOTTOM:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        # if KIDD_NORM_DUP:
        dup_space = m0_raw / metric_norm #.norm(2, -1, True)
        # else:
        #     dup_space = m1_raw
        tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        if KIDD_RND_DUP:
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) #(dup_space @ tokens_pivot.transpose(-2, -1)).sum(-1)
            # score_dup = score_dup.scatter(-1, idx_pivot, -torch.inf)

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        left_idx = idx_tmp[:, r:]

        # src_idx = score_dup.topk(r, sorted=False).indices
        # left_idx = setdiff_indices(seq, src_idx)

        # find merging target basing on similarity, again, with whole left set
        # if KIDD_NORM_MRG:
        #     merge_space = m2_raw / m2_raw.norm(2, -1, True)
        # else:
        #     merge_space = m2_raw

        merge_space = dup_space

        tokens_src = merge_space.gather(-2, src_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        tokens_left = merge_space.gather(-2, left_idx[..., None].expand(-1, -1, merge_space.size(-1)))
        idx_sim = (tokens_left @ tokens_src.transpose(-2, -1)).argmax(dim=-2) #.argmax(-1)
        # assert idx_sim.size(1) == r
        tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m0_prot[:, 0:1]
        else:
            tokens_base = m0_raw.mean(-2, True) # (bsz, 1, dim)
        
        # if KIDD_NORM_IMP:
        #     imp_space = m3_raw / m3_raw.norm(2, -1, True)
        # else:
        #     imp_space = m3_raw
        imp_space = dup_space
        tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        if KIDD_RND_IMP:
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1) # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        # mask_imp = torch.zeros_like(score_imp, dtype=torch.bool)
        # mask_imp.scatter_(-1, idx_imp, True)
        # mask_imp = mask_imp.gather(-1, src_idx)
        # mask_imp_1 = torch.isin(src_idx, idx_imp)
        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)

        # print(torch.allclose(mask_imp, mask_imp_2))

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        tar_idx = tar_idx.where(mask_imp, src_idx)

        # shrunk the array from seq to min(num_imp, r)
        # idx_imp_dup = mask_imp.to(dtype=torch.int).topk(num_imp_dup, sorted=False).indices

        # tar_idx = tar_idx.gather(-1, idx_imp_dup)
        # src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape

        if True:
            src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
            tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
            x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
            left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)
        else:
            x_raw = x_raw.transpose(-2, -1).contiguous() # now it is bsz, dim, seq
            src = x_raw.take_along_dim(src_idx.unsqueeze(-2), -1) # * mask_imp.unsqueeze(-2)

            # Version 0
            if True: #reduce != 'sum':
                tar_idx_ = tar_idx.unsqueeze(-2).expand(-1, dim, -1)
                x_raw = x_raw.scatter_reduce(-1, tar_idx_, src, reduce)
            # Version 1
            else:
                batch_offsets = torch.arange(bsz, device=tar_idx.device, dtype=torch.long).view(bsz, 1, 1) * (dim * seq)
                dim_offsets = torch.arange(dim, device=tar_idx.device, dtype=torch.long).view(1, dim, 1) * seq
                flat_idx = (tar_idx.unsqueeze(-2) + batch_offsets + dim_offsets).flatten()
                flat_src = src.flatten()
                flat_x_raw = x_raw.flatten()
                flat_x_raw = flat_x_raw.index_add(0, flat_idx, flat_src)
                x_raw = flat_x_raw.view(bsz, dim, seq)

            left = x_raw.take_along_dim(left_idx.unsqueeze(-2), -1)
            left = left.transpose(-2, -1)#.contiguous()

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    # slerp
    # size, size_ = handle_size(pinfo, x, merge)
    # return merge(x * size) / size_

    # mlerp
    length = x.norm(2, -1, True)
    length_ = merge(length.view(dtype=torch.int32), 'amax').view(dtype=torch.float32)
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

def kidd_left2_pte(pinfo, cinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, m4 : torch.Tensor, spliter : Spliter):

    m0_prot, m0_raw = spliter(m0)
    m1_prot, m1_raw = spliter(m1)
    m2_prot, m2_raw = spliter(m2)
    m3_prot, m3_raw = spliter(m3)
    m4_prot, m4_raw = spliter(m4)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in cinfo:
        imp_num = cinfo['imp_num']
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in cinfo:
        imp_factor = cinfo['imp_factor']
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in cinfo:
        pivot_num = cinfo['pivot_num']
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in cinfo:
        pivot_factor = cinfo['pivot_factor']
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil(seq * pivot_factor), 1, seq)

    with torch.no_grad():

        # scale the metric matrix
        metric_norm = m0_raw.norm(2, -1, True) # (bsz, seq, 1)
        if KIDD_PIVOT_BOTTOM:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=False).indices
        else:
            idx_pivot = metric_norm[..., 0].topk(num_pivot, sorted=False, largest=True).indices

        # calculate redundancy
        # select pivot tokens, there will be (bsz, num_pivot) indices
        if KIDD_NORM_DUP:
            dup_space = m1_raw / m1_raw.norm(2, -1, True)
        else:
            dup_space = m1_raw
        tokens_pivot = dup_space.gather(-2, idx_pivot.unsqueeze(-1).expand(-1, -1, dup_space.size(-1)))
        if KIDD_RND_DUP:
            score_dup = torch.rand(bsz, seq, device=x.device)
        else:
            score_dup = matmul_sum(dup_space, tokens_pivot) #(dup_space @ tokens_pivot.transpose(-2, -1)).sum(-1)

        # regard top r tokens as duplicate tokens
        idx_tmp = score_dup.sort(descending=True).indices
        src_idx = idx_tmp[:, :r]
        left_idx = idx_tmp[:, r:]

        # find merging target basing on similarity, again, with whole left set
        if KIDD_NORM_MRG:
            merge_space = m2_raw / m2_raw.norm(2, -1, True)
        else:
            merge_space = m2_raw

        tokens_src = merge_space.take_along_dim(src_idx.unsqueeze(-1), -2)
        tokens_left = merge_space.take_along_dim(left_idx.unsqueeze(-1), -2)

        pte_space = pinfo['pte']
        pte_space = pte_space / pte_space.norm(p=2, dim=-1, keepdim=True)
        pte_src = pte_space.take_along_dim(src_idx.unsqueeze(-1), -2)
        pte_left = pte_space.take_along_dim(left_idx.unsqueeze(-1), -2)

        # _, idx_sim = matmul_max(tokens_src, tokens_left) #.argmax(-1)
        _, idx_sim = (tokens_src @ tokens_left.transpose(-2, -1) + pte_src @ pte_left.transpose(-2, -1)).max(-1) #.argmax(-1)
        tar_idx = left_idx.gather(-1, idx_sim)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m3_prot[:, 0:1]
        else:
            tokens_base = m3_raw.mean(-2, True) # (bsz, 1, dim)
        
        if KIDD_NORM_IMP:
            imp_space = m4_raw / m4_raw.norm(2, -1, True)
            tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        else:
            imp_space = m4_raw
        # tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        if KIDD_RND_IMP:
            score_imp = torch.rand(bsz, seq, device=x.device)
        else:
            score_imp = (imp_space @ tokens_base.transpose(-2, -1)).squeeze(-1) # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        mask_imp = (src_idx.unsqueeze(-1) == idx_imp.unsqueeze(-2)).any(-1)

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        tar_idx = tar_idx.where(mask_imp, src_idx)

    def merge(x : torch.Tensor, reduce = 'sum', do_split = True):
        if do_split:
            x_prot, x_raw = spliter(x)
        else:
            x_prot, x_raw = x[:, 0:0, :], x
        bsz, seq, dim = x_raw.shape

        src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        tar_idx_ = tar_idx.unsqueeze(-1).expand(-1, -1, dim)
        x_raw = x_raw.scatter_reduce(-2, tar_idx_, src, reduce)
        left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

        return torch.cat([x_prot, left], dim = -2)
    
    handle_source(pinfo, x, merge)

    # pte = pinfo['pte']
    pinfo['pte'] = merge(pinfo['pte'], do_split=False)

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

def kidd_pivot3m(pinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, spliter : Spliter):

    m0_prot, m0_raw = spliter(m0)
    m1_prot, m1_raw = spliter(m1)
    m2_prot, m2_raw = spliter(m2)

    use_cls = pinfo.get('use_cls', False)
    # pivot_factor = pinfo.get('pivot_factor', [0.05]).pop(0)

    # cal improtance by attention score with mean
    bsz, seq, dim = m0_raw.shape

    if pinfo['tome_scheme']:
        r = min(r, seq // 2)

        if r <= 0:
            return x

    num_imp = max(r, seq - r)
    num_imp_dup = min(num_imp, r)
    num_pivot = math.ceil((seq - r) * 0.05)

    if 'imp_num' in pinfo:
        imp_num = pinfo['imp_num'].pop(0)
        if imp_num is not None:
            num_imp = imp_num
    elif 'imp_factor' in pinfo:
        imp_factor = pinfo['imp_factor'].pop(0)
        if imp_factor is not None:
            num_imp = clamp(math.floor((seq + 1) * imp_factor), 0, seq)
    if 'pivot_num' in pinfo:
        pivot_num = pinfo['pivot_num'].pop(0)
        if pivot_num is not None:
            num_pivot = pivot_num
    elif 'pivot_factor' in pinfo:
        pivot_factor = pinfo['pivot_factor'].pop(0)
        if pivot_factor is not None:
            num_pivot = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)

    with torch.no_grad():
        # select pivot tokens, there will be (bsz, num_pivot) indices
        metric_norm = m0_raw.norm(2, -1) # (bsz, seq, 1)
        idx_pivot = metric_norm.topk(num_pivot, sorted=False).indices
        idx_non_pivot = setdiff_indices(seq, idx_pivot)

        # define the space on which duplication is judged
        if m0 is m1:
            dup_space = m0_raw / metric_norm.unsqueeze(-1)
        else:
            dup_space = m1_raw / m1_raw.norm(2, -1, True)
        tokens_pivot = dup_space.take_along_dim(idx_pivot.unsqueeze(-1), -2)
        tokens_non_pivot = dup_space.take_along_dim(idx_non_pivot.unsqueeze(-1), -2)

        del metric_norm
        del dup_space
        # metric_norm = None

        # calculate redundancy
        # score_dup, idx_dup_tar = (m_raw @ tokens_pivot.transpose(-2, -1)).max(-1)
        score_dup, idx_dup_tar = matmul_max(tokens_non_pivot, tokens_pivot)
        # idx_dup_tar = idx_dup_tar.to(dtype=torch.long)

        # score_tmp = torch.empty((bsz, seq), device=score_dup.device, dtype=score_dup.dtype).

        # regard top r tokens as duplicate tokens
        # idx_tmp = score_dup.scatter(-1, idx_pivot, -torch.inf).sort(descending=True).indices
        # idx_tmp = score_dup.sort(descending=True).indices
        idx_tmp = score_dup.topk(r, 1, True, False).indices
        # idx_dup = idx_non_pivot.gather(-1, score_dup.topk(r, -1, True, False).indices)
        # idx_dup = idx_tmp[:, :r]
        # left_idx = idx_tmp[:, r:].sort().values
        # left_idx = setdiff_indices(seq, )

        # now we assign tar_idx and src_idx, note that they are not filtered by importance yet
        tar_idx = idx_pivot.gather(-1, idx_dup_tar.gather(-1, idx_tmp))
        src_idx = idx_non_pivot.gather(-1, idx_tmp)
        left_idx = setdiff_indices(seq, src_idx)

        # calculate importance now
        if use_cls and pinfo['class_token']:
            tokens_base = m2_prot[:, 0:1]
        else:
            tokens_base = m2_raw.mean(-2, True) # (bsz, 1, dim)
        tokens_base = tokens_base / tokens_base.norm(2, -1, True)
        score_imp = ((m2_raw / m2_raw.norm(2, -1, True)) @ tokens_base.view(bsz, m2.size(-1), 1))[:, :, 0] # (bsz, seq)
        idx_imp = score_imp.topk(num_imp, -1, True, False).indices

        mask_imp = torch.zeros_like(score_imp).scatter(-1, idx_imp, 1.)
        # mask_imp.scatter_(-1, idx_imp, 0)
        mask_imp = mask_imp.gather(-1, src_idx)

        # now only those important tokens have a target, others should be assigned to indices which are not in left_idx
        # tar_idx[~mask_imp] = src_idx[~mask_imp]
        tar_idx = tar_idx.where(mask_imp > 0, src_idx)

        # shrunk the array from seq to min(num_imp, r)
        # idx_imp_dup = mask_imp.topk(num_imp_dup, sorted=False).indices

        # tar_idx = tar_idx.gather(-1, idx_imp_dup)
        # src_idx = src_idx.gather(-1, idx_imp_dup)

    def merge(x : torch.Tensor, reduce = 'sum'):
        x_prot, x_raw = spliter(x)
        bsz, seq, dim = x_raw.shape

        src = x_raw.gather(-2, src_idx.unsqueeze(-1).expand(-1, -1, dim))
        # src = x_raw.take_along_dim(src_idx.unsqueeze(-1), -2)
        x_raw = x_raw.scatter_reduce(-2, tar_idx.unsqueeze(-1).expand(-1, -1, dim), src, reduce)
        left = x_raw.gather(-2, left_idx.unsqueeze(-1).expand(-1, -1, dim))
        # left = x_raw.take_along_dim(left_idx.unsqueeze(-1), -2)

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