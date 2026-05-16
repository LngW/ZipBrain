import os
import math
import torch

from .utils import separate_method_args, Spliter, handle_source, clamp, setdiff_indices

KIDD_RND_DUP=(os.getenv('KIDD_RND_DUP', '0') == '1')
KIDD_RND_IMP=(os.getenv('KIDD_RND_IMP', '0') == '1')
KIDD_NORM_DUP=(os.getenv('KIDD_NORM_DUP', '1') == '1')
KIDD_NORM_MRG=(os.getenv('KIDD_NORM_MRG', '1') == '1')
KIDD_NORM_IMP=(os.getenv('KIDD_NORM_IMP', '1') == '1')

KIDD_PIVOT_BOTTOM=(os.getenv('KIDD_PIVOT_BOTTOM', '0') == '1')

# In this method, we return three metric, and the three metrics are used for:
# m0: select pivot tokens
# m1: decide the space of judging duplication
# m2: device the space of merging target
# m3: decide the space of judging importance
# We accept three formats: [m0_1_2_3], [m0_1_2, m3], [m0, m1_2, m3] or [m0, m1, m2, m3]
# When not explictly indicated, we use k for m0, m1, m2 and m3
def select_metric(args : list[str], x, q, k, v):
    # left = variant.find('[')
    # right = variant.find(']')

    # if left <= 0 or right <= 0 or right <= left:
    #     return k, k, k, k
    
    # scheme = variant[left + 1:right]
    # if len(scheme) <= 0:
    #     return k, k, k, k
    
    # scheme = scheme.split(',')
    # scheme = [it.strip() for it in scheme]
    # args = args
    mapping = {'x': x, 'q': q, 'k': k, 'v': v, 'r': torch.rand_like(k)}
    if len(args) <= 0:
        return k, k, k, k
    elif len(args) == 1:
        m0 = mapping.get(args[0], k)
        return m0, m0, m0, m0
    elif len(args) == 2:
        m0_1_2 = mapping.get(args[0], k)
        m3 = mapping.get(args[1], k)
        return m0_1_2, m0_1_2, m3
    elif len(args) == 3:
        m0 = mapping.get(args[0], k)
        m1_2 = mapping.get(args[1], k)
        m3 = mapping.get(args[2], k)
        return m0, m1_2, m1_2, m3
    else:
        return [mapping.get(it, k) for it in args[:4]]


def apply_kidd(pinfo : dict, r : int, variant : str, x : torch.Tensor, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):

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
    
    method, args = separate_method_args(variant)

    # print(args)

    m0, m1, m2, m3 = select_metric(args, x, q.mean(1), k.mean(1), v.mean(1)) # q, k, v are all (B, N, HD) now, and x is (B, N, D)

    if method == 'kidd3mp':
        return kidd_pivot3m(pinfo, r_, x, m0, m1, m2, split)
    elif method == 'kiddp':
        return kidd_pivot(pinfo, r_, x, m0, split)
    elif method == 'kiddl4':
        return kidd_left4(pinfo, r_, x, m0, split)
    elif method == 'kiddl3':
        return kidd_left3(pinfo, r_, x, m0, m1, m2, m3, split)
    elif method == 'kiddl2pte':
        return kidd_left2_pte(pinfo, r_, x, m0, m1, m2, m3, split)
    elif method == 'kiddl2':
        return kidd_left2(pinfo, r_, x, m0, m1, m2, m3, split)
    elif method == 'kiddl':
        return kidd_left(pinfo, r_, x, m0, split)
    else:
        raise NotImplementedError('Unsupported KIDD variant or args: {}, {}'.format(method, args))


    if variant.startswith('kidd3m'):
        if variant.startswith('kidd3mp'):
            return kidd_pivot3m(pinfo, r_, x, m0, m1, m2, split)
        else:
            raise NotImplementedError('Unsupported KIDD variant')
    elif variant.startswith('kidd'):
        if variant.startswith('kiddp'):
            return kidd_pivot(pinfo, r_, x, m0, split)
        elif variant.startswith('kiddl4'):
            return kidd_left4(pinfo, r_, x, m0, split)
        elif variant.startswith('kiddl3'):
            return kidd_left3(pinfo, r_, x, m0, m1, m2, m3, split)
        elif variant.startswith('kiddl2pte'):
            return kidd_left2_pte(pinfo, r_, x, m0, m1, m2, m3, split)
        elif variant.startswith('kiddl2'):
            return kidd_left2(pinfo, r_, x, m0, m1, m2, m3, split)
        elif variant.startswith('kiddl'):
            return kidd_left(pinfo, r_, x, m0, split)
        else:
            raise NotImplementedError('Unsupported KIDD variant')

def matmul_sum(a : torch.Tensor, b : torch.Tensor):
    return (a @ b.transpose(-2, -1)).sum(-1)

def matmul_max(a : torch.Tensor, b : torch.Tensor):
    return (a @ b.transpose(-2, -1)).max(-1)

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

def kidd_pivot(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

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

def kidd_left(pinfo, r : int, x : torch.Tensor, metric : torch.Tensor, spliter : Spliter):

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

def kidd_left2(pinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, spliter : Spliter):

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

def kidd_left2_pte(pinfo, r : int, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, m2 : torch.Tensor, m3 : torch.Tensor, spliter : Spliter):

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
            imp_space = m3_raw / m3_raw.norm(2, -1, True)
        else:
            imp_space = m3_raw
        tokens_base = tokens_base / tokens_base.norm(2, -1, True)
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
