
def create_biot_encoder_block(
    dim, 
    heads, 
    depth, 
    max_seq_len, 
    attn_layer_dropout, 
    attn_dropout, 
    **kwargs):

    variants = kwargs.get('variants', '').lower()

    if variants == 'topk':
        from .blocks import TopKEncoder
        return TopKEncoder(
            emb_size=dim,
            heads=heads,
            k = kwargs['k'],
            ffn_hidden_size=dim * 4,
            num_layers=depth,
            dropout=attn_dropout
        )
    elif variants == 'tome':
        from .blocks import ToMeEncoder
        return ToMeEncoder(
            emb_size=dim,
            heads = heads,
            r = kwargs['r'],
            ffn_hidden_size=dim * 4,
            num_layers=depth,
            dropout=attn_dropout
        )
    elif variants == 'custom':
        from .blocks2 import LinearAttentionEncoder
        return LinearAttentionEncoder(
            dim=dim,
            heads=heads,
            depth=depth,
            ff_dropout=0,
            attn_dropout=attn_dropout
        )
    elif variants == 'ctome':
        from .blocks2 import ToMeEncoder
        return ToMeEncoder(
            emb_size=dim,
            heads=heads,
            r = kwargs['r'],
            num_layers=depth,
            dropout=attn_dropout
        )
    else:
        from linear_attention_transformer import LinearAttentionTransformer
        return LinearAttentionTransformer(
            dim = dim,
            heads = heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout = attn_layer_dropout,
            attn_dropout = attn_dropout,
        )
    
    