
def apply_merge(pinfo, r : int, variant : str, x):

    q, k, v = pinfo['qkv']
    pinfo['qkv'] = None

    if r <= 0:
        return x

    fast_dict = {'q': q, 'k': k, 'v': v, 'x': x}
    def _find_metric(variant):
        left = variant.find('[')
        right = variant.find(']')
        if left > 0 and right > left:
            return fast_dict.get(variant[left + 1 : right], k)
        else:
            return k
    
    if variant is not None:
        metric = _find_metric(variant)

    from merges import apply_merge_impl
    return apply_merge_impl(pinfo, r, variant, x, metric)

def make_sequential_class(klass):
    class PatchedSequentialSequence(klass):
        def forward(self, x, **kwargs):
            from linear_attention_transformer.reversible import route_args, layer_drop
            args = route_args(self.args_route, kwargs, len(self.layers))

            # modification start
            layers_and_args = list(zip(self.layers, args, self._pinfo['r'], self._pinfo['variant']))
            # r = self._pinfo['r'].pop(0)
            # variant = self._pinfo['variant'].pop(0)
            # modification end

            if self.training and self.layer_dropout > 0:
                layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

            # modified this line: add r and variant
            for (f, g), (f_args, g_args), r, variant in layers_and_args:
                if self._pinfo['show_shape']: print(x.shape, self._pinfo['class_token'])
                x = x + f(x, **f_args)
                # insertation here
                x = apply_merge(self._pinfo, r, variant, x)
                # insertation end
                x = x + g(x, **g_args)
            return x
        
    return PatchedSequentialSequence

def make_self_attention_class(klass):

    class PatchedSelfAttention(klass):
        def forward(self, x, input_mask = None, context = None, context_mask = None, pos_emb = None, **kwargs):
            from linear_attention_transformer.linear_attention_transformer import exists, apply_rotory_pos_emb, split_at_index
            import torch
            from functools import partial

            assert not (self.receives_context and not exists(context)), 'context must be supplied if self attention is in receives context mode'

            if not self.receives_context:
                q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
            else:
                q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

            b, t, e, h, dh = *q.shape, self.heads, self.d_heads

            merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

            q, k, v = map(merge_heads, (q, k, v))

            if exists(pos_emb) and not self.receives_context:
                q, k = apply_rotory_pos_emb(q, k, pos_emb)

            out = []

            split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

            (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

            has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

            if has_local:
                local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
                out.append(local_out)

            if has_global:
                kv_mask = input_mask if not self.receives_context else context_mask
                global_out = self.global_attn_fn(q, k, v, kv_mask = kv_mask)
                out.append(global_out)

            attn = torch.cat(out, dim=1)
            attn = attn.transpose(1, 2).reshape(b, t, -1)

            # modification
            self._pinfo['qkv'] = [q.mean(1), k.mean(1), v.mean(1)]
            # modification End

            return self.dropout(self.to_out(attn))
    
    return PatchedSelfAttention
