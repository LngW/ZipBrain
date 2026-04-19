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

    if variant.startswith('dartpa'):
        return dart_pruneA(pinfo, r_, x, m0, split)
    if variant.startswith('dartp'):
        return dart_prune(pinfo, r_, x, m0.detach(), m1.detach(), split)
    else:
        return x

def dart_prune(pinfo, r, x : torch.Tensor, m0 : torch.Tensor, m1 : torch.Tensor, split : Spliter):

    m0_prot, m0_raw = split(m0)
    m1_prot, m1_raw = split(m1)

    bsz, seq, dim = m1_raw.shape

    pivot_factor = pinfo.get('pivot_factor', [None]).pop(0)
    if pivot_factor is None:
        pivot_factor = 0.05

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