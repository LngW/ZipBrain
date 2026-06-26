
import torch

def apply_collect(pinfo, cinfo, r, variant, x, q, k, v):
    collects = pinfo['collect'] = pinfo.get('collect', {})
    attns = collects['attns'] = collects.get('attns', [])
    attns.append((q @ k.transpose(-1, -2)).softmax(dim=-1))

    if pinfo.get('class_token', False):
        q_ = q[..., 1:, :]
        k_ = k[..., 1:, :]
    else:
        q_ = q
        k_ = k

    means = collects['means'] = collects.get('means', [])
    means.append((q_.mean(-2, True) @ k_.transpose(-1, -2)).detach())

    sqks = collects['sqks'] = collects.get('sqks', [])
    sqks.append((q.softmax(dim=-1) @ k.transpose(-1, -2).softmax(dim=-1)).detach())

    sqks2 = collects['sqks2'] = collects.get('sqks2', [])
    sqks2.append((q.softmax(dim=-1) @ k.softmax(dim=-1).transpose(-1, -2)).detach())

    xqkv = collects['xqkv'] = collects.get('xqkv', [])
    xqkv.append((x.detach(), q.detach(), k.detach(), v.detach()))

    return x