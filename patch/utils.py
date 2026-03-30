def reset_common_pinfo(model, pinfo, depth):
    r = getattr(model, 'r', 0)
    variant = getattr(model, 'variant', '')

    pinfo['r'] = parse_r(depth, r)
    pinfo['variant'] = parse_variant(depth, variant)

    def optional_set(key):
        if hasattr(model, key):
            pinfo[key] = repeat_or_pad(getattr(model, key), depth, None)

    optional_set('pivot_factor')
    optional_set('pivot_num')
    optional_set('imp_factor')
    optional_set('imp_num')

def repeat_or_pad(it, depth, pad):
    if not isinstance(it, list):
        return [it] * depth
    
    size = len(it)

    if size == 1:
        return it * depth
    
    return it + [pad] * (depth - size)

def parse_r(depth, r):
    if r is None:
        return [0] * depth
    
    if isinstance(r, int):
        return [r] * depth

    assert isinstance(r, list)

    size = len(r)
    if size == 1:
        r : str | int = r[0]

        if isinstance(r, int):
            r = [r] * depth
        else:
            if r.startswith('L/'):
                r = [int(r[2:])]
            elif r.startswith('F/'):
                base, factor = r[2:].split(',')
                base = int(base)
                factor = float(factor)

                r = [base]
                for it in range(depth - 1):
                    r.append(r[it] * factor)
            else:
                r = [int(r)] * depth
    elif size > 1:
        r = [int(it) for it in r]

    size = len(r)
    r = r + [0] * (depth - size)

    return r

def parse_variant(depth, variant):
    if variant is None:
        return [''] * depth
    
    if isinstance(variant, str):
        return [variant] * depth

    assert isinstance(variant, list)

    size = len(variant)
    if size == 1:
        variant : str = variant[0]

        if variant.startswith('L/'):
            variant = [variant[2:]]
        else:
            variant = [variant] * depth
    
    size = len(variant)
    variant = variant + [''] * (depth - size)

    return variant
