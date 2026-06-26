import math
from typing import Callable
import torch
from einops import rearrange
from .utils import separate_method_args, clamp, handle_size, handle_source, Spliter

def find_two_metric(
        args : list[str], 
        x: torch.Tensor, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ):
    # left = variant.find('[')
    # right = variant.find(']')

    # if left < 0 or right < 0 or right <= left:
    #     return q, k

    # sub = variant[left + 1:right]
    # subs = [it.strip() for it in sub.split(',')]

    subs = args
    if len(subs) == 0:
        return q, k
    
    table = {'x': x, 'q': q, 'k': k, 'v': v}
    if len(subs) == 1:
        m = table.get(subs[0], q)

        return m, m
    
    metrics = [table.get(it, q) for it in subs]

    return metrics[0], metrics[1]

def apply_dart(pinfo : dict, cinfo : dict, r : int, variant : str, x : torch.Tensor, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
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
    m0, m1 = find_two_metric(
        args, 
        x, 
        rearrange(q, 'b h n d -> b n (h d)'), 
        rearrange(k, 'b h n d -> b n (h d)'), 
        rearrange(v, 'b h n d -> b n (h d)')
    )

    if method == 'dartpo':
        return original_dartp_vectorized(pinfo, cinfo, r_, x, m0, split)
    elif method == 'dartpa':
        return dart_pruneA(pinfo, r_, x, m0, split)
    elif method == 'dartp':
        m0, m1 = find_two_metric(
            args, 
            x, 
            q.mean(1),
            k.mean(1),  
            v.mean(1), 
        )
        return dart_prune(pinfo, cinfo, r_, x, m0.detach(), m1.detach(), split)
    elif method == 'dartp_':
        m0, m1 = find_two_metric(
            args, 
            x, 
            rearrange(q, 'b h n d -> b n (h d)'), 
            rearrange(k, 'b h n d -> b n (h d)'), 
            rearrange(v, 'b h n d -> b n (h d)')
        )
        return dart_prune(pinfo, cinfo, r_, x, m0.detach(), m1.detach(), split)
    else:
        raise NotImplementedError("Unknown dart method: {} {}".format(method, args))

def dart_prune(pinfo, cinfo, r, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, split : Spliter):

    m0_prot, m0_raw = split(m0)
    m1_prot, m1_raw = split(m1)

    bsz, seq, dim = m1_raw.shape

    pivot_factor = cinfo.get('pivot_factor', None)
    if pivot_factor is None:
        pivot_factor = 0.05

    k = clamp(math.ceil((seq - r) * pivot_factor), 1, seq - r)
    # select tokens via m0's l1-norm
    pivot_idx = m0_raw.norm(dim=-1, p=1).topk(k=k, dim=-1).indices
    # but test token's redundancy via m1's cosine similarity
    pivot_tokens = m1_raw.take_along_dim(pivot_idx[..., None], -2)

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


def __dart_impl0(config, r, x, metric, split):
    # image_token_start_index = config['image_token_start_index']
    # image_token_length = config['image_token_length']

    # device = metric.device

    retained_image_tokens_index = get_retained_image_token_(config, r, x, metric)

    keep_indexs = retained_image_tokens_index # torch.cat([
        # torch.arange(image_token_start_index,device=device), 
#        retained_image_tokens_index, 
        # torch.arange(image_token_start_index+image_token_length, seq_length, device=device)
#    ])
    keep_indexs = keep_indexs.sort().values

    # print(keep_indexs.shape[-1])

    def merge(x, mode = ''):
        x_prot, x_raw = split(x)
        x_raw = torch.take_along_dim(x_raw, keep_indexs.unsqueeze(-1).unsqueeze(0), -2)

        return torch.cat([x_prot, x_raw], dim=-2)

    # hidden_states = x[:,keep_indexs,:]
    # if attention_mask is not None:
    #     attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
    # position_ids = keep_indexs.unsqueeze(0)

    handle_size(config, x, merge)
    handle_source(config, x, merge)

    return merge(x)

# def get_retained_image_token_(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
def get_retained_image_token_(config: dict, r, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
    # DART_config = config.DART_config
    DART_config = config
    # K = DART_config['K']  # pruned layer
    # image_token_start_index = DART_config['image_token_start_index']
    # image_token_length = DART_config['image_token_length']
    # MAX_NUM_TRUNCTION = DART_config['max_num_trunction']
    image_token_start_index = 0 # DART_config['image_token_start_index']
    image_token_length = last_layer_state.size(-2) # DART_config['image_token_length']
    MAX_NUM_TRUNCTION = None # DART_config['max_num_trunction']

    # pivot_image_token = DART_config['pivot_image_token']
    # pivot_text_token = DART_config['pivot_text_token']
    pivot_factor = config.get('pivot_factor', [None]).pop(0)
    if pivot_factor is None:
        pivot_factor = 0.05

    pivot_image_token = math.ceil(pivot_factor * image_token_length) # DART_config['pivot_image_token']
    pivot_text_token = 0 # DART_config['pivot_text_token']

    # reduction_ratio = DART_config['reduction_ratio']
    reduction_ratio = r / image_token_length
    TOKEN_TOPK = math.ceil((MAX_NUM_TRUNCTION if MAX_NUM_TRUNCTION is not None else (image_token_length * (1 - reduction_ratio))) // (pivot_image_token + pivot_text_token))

    device = last_layer_state.device

    # if config.text_length is not None:
    #     text_length = config.text_length
    #     image_token_length = any_states.shape[2] - text_length
    #     retain_token_num_for_llava_next = min(DART_config['retain_token_num_for_llava_next'], image_token_length - pivot_image_token)
    #     TOKEN_TOPK = int((retain_token_num_for_llava_next if retain_token_num_for_llava_next is not None else (image_token_length * (1 - reduction_ratio)))  // (pivot_image_token + pivot_text_token))

    # any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.shape[0], any_states.shape[2], -1)

    k_states_image_token = any_states[0][image_token_start_index:image_token_start_index + image_token_length, :]
    # k_states_query_token = any_states[0][image_token_start_index + image_token_length:, :]

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    # k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

    image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist() 
    # query_indices = (k_states_query_token_L1_norm.topk(pivot_text_token).indices + image_token_start_index + image_token_length).tolist()
    # indices_set = set(image_indices + query_indices)
    indices_set = set(image_indices)

    valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)

    valid_indices_list = list(valid_indices)  
    for item in list(indices_set):
        valid_vectors = last_layer_state[0][valid_indices_list, :]
        cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[0][item, :], valid_vectors, dim=-1)
        if cos_sim.size(-1) <= TOKEN_TOPK:
            top_k_indices = torch.arange(cos_sim.size(-1), dtype=torch.long, device=cos_sim.device)
        else:
            top_k_indices = cos_sim.topk(TOKEN_TOPK).indices

        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
        indices_set.update(top_k_real_indices)
        
        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)  

    # indices_set.difference_update(query_indices)

    retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

    return retained_image_tokens_index


def dart_pruneA(pinfo, r, x, metric : torch.Tensor, split):
    return __dart_impl0(pinfo, r, x, metric, split)

# def original_dartp(
#     pinfo : dict,
#     r : int,
#     x: torch.Tensor,
#     m: torch.Tensor,
#     # DART_config: dict
#     spliter,
# ) -> torch.Tensor:
#     """
#     Simplified version of DART token reduction, assuming NO text tokens (only image tokens).
    
#     Args:
#         last_layer_state (torch.Tensor): Hidden states from the last layer (e.g., [1, seq_len, hidden_dim]).
#         any_states (torch.Tensor): States from the pruned layer (e.g., [1, seq_len, any_dim]).
#         DART_config (dict): Configuration containing 'image_token_start_index', 'image_token_length',
#                             'pivot_image_token', 'reduction_ratio', and optionally 'max_num_trunction'.
    
#     Returns:
#         torch.Tensor: Tensor containing indices of the retained image tokens.
#     """

#     x_prot, x_raw = spliter(x)
#     x_prot : torch.Tensor = x_prot
#     x_raw : torch.Tensor = x_raw

#     m_prot, m_raw = spliter(m)

#     image_token_start_index = 0 #DART_config['image_token_start_index']
#     image_token_length = x_raw.shape[1] #DART_config['image_token_length']
#     MAX_NUM_TRUNCTION = None #DART_config.get('max_num_trunction')
#     pivot_image_token = max(math.floor(image_token_length * pinfo.get('pivot_factor', [0.05]).pop(0)), 1)
#     reduction_ratio = r / image_token_length #DART_config['reduction_ratio']
    
#     device = x_raw.device
    
#     # Calculate TOKEN_TOPK based only on image tokens
#     token_topk = math.ceil(
#         (MAX_NUM_TRUNCTION if MAX_NUM_TRUNCTION is not None else (image_token_length * (1 - reduction_ratio))) 
#         // pivot_image_token
#     )
    
#     # Extract image token states
#     k_states_image_token = m_raw[0] #any_states[0, image_token_start_index : image_token_start_index + image_token_length, :]
#     k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    
#     # Select pivot tokens
#     image_indices = (k_states_image_token_L1_norm.topk(pivot_image_token).indices + image_token_start_index).tolist() 
#     indices_set = set(image_indices)
    
#     valid_indices = set(range(image_token_start_index, image_token_start_index + image_token_length)) - set(image_indices)
    
#     valid_indices_list = list(valid_indices)  
#     for item in list(indices_set):
#         if not valid_indices_list: break
        
#         valid_vectors = x_raw[0, valid_indices_list, :]
#         cos_sim = -torch.nn.functional.cosine_similarity(x_raw[0, item, :], valid_vectors, dim=-1)
        
#         # Ensure token_topk does not exceed number of available valid indices
#         current_k = min(token_topk, len(valid_indices_list))
#         top_k_indices = cos_sim.topk(current_k).indices
        
#         top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
#         indices_set.update(top_k_real_indices)
        
#         valid_indices.difference_update(top_k_real_indices)
#         valid_indices_list = list(valid_indices)  
    
#     x_raw_ = x_raw[0].take_along_dim(torch.tensor(list(indices_set), device=device).unsqueeze(-1), -2)
#     return torch.cat([x_prot, x_raw_.unsqueeze(0)], -2)
#     # return torch.tensor(list(indices_set), device=device)

def original_dartp(
    pinfo : dict,
    cinfo : dict,
    r : int,
    x: torch.Tensor,
    m: torch.Tensor,
    spliter: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
):
    device = x.device
    x_prot, x_raw = spliter(x)
    m_prot, m_raw = spliter(m)

    seq = x_raw.shape[-2]
    # Here we requires at least one pivot is chosen.
    pivot_image_token = max(math.floor(seq * cinfo.get('pivot_factor', 0.05)), 1)

    # Select pivot tokens
    k_norm = torch.norm(m_raw, p=1, dim=-1)
    pivot_indices = k_norm.topk(pivot_image_token).indices

    # From here, we need to proceed sample by sample
    _processed = []
    for b in range(x.shape[0]):
        pivot_indice_list = pivot_indices[b].tolist()
        left_indices_set = set(pivot_indice_list)
    
        valid_indices_set = set(range(seq)) - left_indices_set
        valid_indices_list = list(valid_indices_set)
        _tmp = seq - pivot_image_token - r

        for i, item in enumerate(pivot_indice_list):
            valid_vectors = x_raw[b, valid_indices_list, :]
            cos_sim = -torch.nn.functional.cosine_similarity(x_raw[b, item, :], valid_vectors, dim=-1)

            # We compute a number of tokens to be recalled for each pivot dynamically here.
            # So that the total number of retained tokens are exactly equal to the desired number.
            token_topk = (_tmp + pivot_image_token - i - 1) // (pivot_image_token - i)
            _tmp -= token_topk

            # current_k = min(token_topk, len(valid_indices_list))
            top_k_indices = cos_sim.topk(token_topk).indices
            
            top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
            left_indices_set.update(top_k_real_indices)
            
            valid_indices_set.difference_update(top_k_real_indices)
            valid_indices_list = list(valid_indices_set)
        x_raw_ = x_raw[b].take_along_dim(torch.tensor(list(left_indices_set), device=device).unsqueeze(-1), -2)
        _processed.append(torch.cat([x_prot[b], x_raw_], -2))

    return torch.stack(_processed)

def original_dartp_vectorized(
    pinfo : dict,
    cinfo : dict,
    r : int,
    x: torch.Tensor,
    m: torch.Tensor,
    spliter: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
):
    device = x.device
    x_prot, x_raw = spliter(x)
    m_prot, m_raw = spliter(m)

    bsz, seq, dim = x_raw.shape

    # Here we requires at least one pivot is chosen.
    pivot_num = max(math.floor(seq * cinfo.get('pivot_factor', 0.05)), 1)
    token_topks = []
    _tmp = seq - pivot_num - r
    for i in range(pivot_num):
        token_topk = (_tmp + pivot_num - i - 1) // (pivot_num - i)
        _tmp -= token_topk
        token_topks.append(token_topk)
    
    del _tmp


    # Select pivot tokens
    k_norm = torch.norm(m_raw, p=1, dim=-1)
    pivot_indices = k_norm.topk(pivot_num).indices

    x_raw_norm = torch.nn.functional.normalize(x_raw, dim=-1)
    pivot_vectors = torch.gather(
        x_raw, 1, pivot_indices.unsqueeze(-1).expand(-1, -1, dim)
    )
    pivot_vectors_norm = torch.nn.functional.normalize(
        pivot_vectors, dim=-1
    )

    # 矩阵乘法代替循环计算相似度
    cos_sim_matrix = -torch.bmm(
        pivot_vectors_norm, x_raw_norm.transpose(1, 2)
    )

    # 3. 构造初始掩码 (把初始的 pivot 标记为不可选 True)
    mask = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    mask.scatter_(1, pivot_indices, True)

    # 记录所有被选中的索引
    all_selected_indices = [pivot_indices]

    # 4. 纯 GPU 内部的顺序掩码迭代 (无 CPU 同步开销)
    for i in range(pivot_num):
        k = token_topks[i]
        if k <= 0:
            continue
        sim_scores = cos_sim_matrix[:, i, :].clone()

        # 核心等价替换：不可选的元素赋为负无穷
        sim_scores.masked_fill_(mask, float("-inf"))

        # 选出当前 pivot 对应的 top-k 真实全局索引
        _, top_k_idx = torch.topk(sim_scores, k=k, dim=-1)
        all_selected_indices.append(top_k_idx)

        # 更新掩码
        mask.scatter_(1, top_k_idx, True)

    # 5. 统一 Gather 并与 x_prot 拼接
    final_indices = torch.cat(all_selected_indices, dim=1)  # (B, total_retained)

    x_raw_processed = torch.gather(
        x_raw, 1, final_indices.unsqueeze(-1).expand(-1, -1, dim)
    )

    # 替代原代码最后一行的 torch.stack(_processed)
    return torch.cat([x_prot, x_raw_processed], dim=-2)

def unit_test_dartpo():
    x = torch.randn(2, 8, 2)
    m = torch.randn(2, 8, 4)
    r = 3

    pinfo = {}
    cinfo = {}

    def split(it):
        return it[:, 0:0], it[:, 0:]
    
    re0 = original_dartp(pinfo, cinfo, r, x, m, split)
    re1 = original_dartp_vectorized(pinfo, cinfo, r, x, m, split)

    re0 = sort_sequences_by_first_component(re0)
    re1 = sort_sequences_by_first_component(re1)

    # print()

    assert torch.allclose(re0, re1)

    pass

def sort_sequences_by_first_component(x, descending=False):
    """根据每个向量的第一个分量大小，对序列中的向量进行排序

    Args:
        x: 输入张量，形状为 (B, N, C) -> (Batch, Sequence_Length, Vector_Dim)
        descending: 是否降序排列。默认 False（升序）
    """
    B, N, C = x.shape

    # 1. 提取所有向量的第一个分量，形状为 (B, N)
    first_component = x[:, :, 0]

    # 2. 获取排序后的索引，形状为 (B, N)
    # 每一行（每个样本）会独立计算自己序列内部的排序索引
    sorted_indices = torch.argsort(first_component, dim=1, descending=descending)

    # 3. 将索引扩展至 (B, N, C)，以便能同时作用于所有的特征通道
    expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, C)

    # 4. 使用 torch.gather 沿着序列维度 (dim=1) 重新排列原张量
    sorted_x = torch.gather(x, dim=1, index=expanded_indices)

    return sorted_x

if __name__ == '__main__':
    unit_test_dartpo()