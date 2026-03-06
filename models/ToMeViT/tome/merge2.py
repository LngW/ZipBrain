import torch

def do_nothing(x, mode = 'mean'):
    return x

def call_variant(variant, x, metric, r, class_token, distill_token):
    if variant == 'f':
        return full_merge(x, metric, r, class_token, distill_token)
    return do_nothing, do_nothing
    # pass

def full_merge(x, metric, r, class_token, distill_token):
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    t = metric.shape[1]
    r = min(r, t - protected)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        q = metric[..., protected:, :]
        q = q / q.norm(dim = -1, keepdim = True)

        similarity = q @ q.transpose(-2, -1)
        importance = torch.exp(similarity).sum(-1)

        _, top_idx = torch.topk(importance, t - r)

        matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, t - protected))
        matrix = (matrix + 1) / 2
        matrix.scatter_(-1, top_idx[..., None], 1)

        matrix_ = torch.zeros_like(matrix)
        matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

        similarity = None
        importance = None
        top_idx = None

        matrix = matrix * matrix_
        matrix = matrix / matrix.sum(-1, True)

    def soft_merge(x : torch.Tensor, mode = 'ignored'):
        return torch.cat([x[..., :protected, :], matrix @ x[..., protected:, :]], dim=-2)

    def hard_merge(x : torch.Tensor, mode = 'ignored'):
        return torch.cat([x[..., :protected, :], matrix_ @ x[..., protected:, :]], dim=-2)
    
    return soft_merge, hard_merge


def _qk_merge0(r, x, q, k, size):
    bsz, seq, dim = x.shape
    r = min(max(r, 0), seq - 1) # keep at least one token

    if r == 0:
        return x, size
    q_only = (q is k) or (k is None)
    if q_only:
        q = q / q.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ q.transpose(-1, -2)
        importance = torch.exp(similarity).sum(-1)
    else:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        similarity : torch.Tensor = q @ k.transpose(-1, -2)
        similarity_ : torch.Tensor = torch.exp(similarity)
        importance = similarity_.sum(-1) - torch.diagonal(similarity_, 0, -2, -1) # so similarity to itself do not affect

    top_v, top_idx = torch.topk(importance, seq - r)

    matrix = similarity.gather(-2, top_idx[..., None].expand(-1, -1, seq))
    matrix = (matrix + 1) / 2
    matrix.scatter_(-1, top_idx[..., None], 1)

    matrix_ = torch.zeros_like(matrix)
    matrix_.scatter_(-2, matrix.argmax(-2, True), 1) # merge to its most similar one

    matrix = matrix * matrix_
    matrix = matrix / matrix.sum(-1, True)

    return matrix @ x, matrix_ @ size, similarity


if __name__ == '__main__':
    x = torch.rand(8, 16, 32)
    soft, hard = full_merge(None, x, 12, True, True)

    soft(x)

    hard(x)