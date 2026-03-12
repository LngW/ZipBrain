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

    from tome.merge_variants import apply_merge as apply_merge0
    return apply_merge0(pinfo, r, variant, x, metric)