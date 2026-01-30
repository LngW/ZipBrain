import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
from torch import Tensor
from torch.nn import functional as F
import math

def ToMeBlock(r = 2, *args, **kwargs):
    from .tome.merge import bipartite_soft_matching, merge_wavg

    class ToMeBlock(nn.Module):
        def __init__(self, r = 2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r = r

        def forward(self, x, k, trace):
            bsz, hd, seq, _ = k.shape
            merge, _ = bipartite_soft_matching(
                k.transpose(1,2).reshape(bsz, seq, -1),
                self.r
            )

            if trace is None:
                trace = torch.eye(seq, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(bsz, 1, 1)

            x = merge(x, mode='sum')
            trace = merge(trace, mode='sum')
            x = x / trace.sum(-1, True)

            return x, trace
        
    return ToMeBlock(r = r, *args, **kwargs)

# Use a customized Multihead attention to integrate ToMe and top-k
class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout, bias, batch_first, device, dtype, 
                 topk = 0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.topk = topk

        self.batch_first = batch_first

        self.q_proj = nn.Linear(d_model, d_model, bias = bias, device = device, dtype = dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias = bias, device = device, dtype = dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias = bias, device = device, dtype = dtype)
        self.dropout = dropout

        self.out_proj = nn.Linear(d_model, d_model)

        pass

    def forward(self, q, k, v, /, attn_mask, key_padding_mask, need_weights):
        n_head = self.nhead
        d_head = self.d_head
        qs = self.q_proj(q).view(*(q.shape[:2]), n_head, d_head).transpose(1, 2)
        ks = self.k_proj(k).view(*(k.shape[:2]), n_head, d_head).transpose(1, 2)
        vs = self.v_proj(v).view(*(v.shape[:2]), n_head, d_head).transpose(1, 2)

        attn_score = (qs @ ks.transpose(-1, -2)) / math.sqrt(d_head)
        if self.topk > 0 and self.topk < q.shape[1]:
            topk, _ = torch.topk(attn_score, self.topk, dim=-1, sorted=False)
            attn_score[attn_score < topk.amin(dim=-1, keepdim = True)] = -torch.inf

        attn_weight = torch.softmax(attn_score, -1)
        attn_weight = torch.dropout(attn_weight, self.dropout, self.training)
        output = (attn_weight @ vs).transpose(1, 2).flatten(-2)

        output = self.out_proj(output)
        return output, ks

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src
        trace_t = None
        trace_s = None
        for mod in self.layers:
            output, trace_t, trace_s = mod(output, trace_t, trace_s, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 top_k_s = 0, top_k_t = 0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = Attention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 topk=top_k_s,
                                                 **factory_kwargs)
        self.self_attn_t = Attention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 topk=top_k_t,
                                                 **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.tome_block_t = ToMeBlock(0)
        self.tome_block_s = ToMeBlock(0)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            trace_t: Tensor,
            trace_s: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x = src
        x1, trace_t, trace_s = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, trace_t=trace_t, trace_s = trace_s)
        x = x + x1
        x = x + self._ff_block(self.norm2(x))
        return x, trace_t, trace_s

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False, trace_t = None, trace_s = None) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz*ch_num, patch_num, patch_size // 2)

        if trace_s is not None:
            xs = (trace_s @ xs) / trace_s.sum(-1, keepdim=True)
        xs, ks = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)
        xs2, trace_s = self.tome_block_s(xs, ks, trace_s)
        xs = trace_s.transpose(-1, -2) @ xs2
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)

        if trace_t is not None:
            xt = (trace_t @ xt) / trace_t.sum(-1, keepdim=True)
        xt, kt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)
        xt2, trace_t = self.tome_block_t(xt, kt, trace_t)
        xt = trace_t.transpose(-1, -2) @ xt2
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x), trace_t, trace_s

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, norm_first=True,
        activation=F.gelu
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
    encoder = encoder.cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()
    b = encoder(a)
    print(a.shape, b.shape)