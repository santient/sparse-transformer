import math

import torch
from torch import nn


class SparseMultiheadAttention(nn.Module):
    """Simple sparse multihead attention using a limited attention span"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, attn_span=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_span = attn_span
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.query_ff = nn.Linear(embed_dim, embed_dim)
        self.key_ff = nn.Linear(embed_dim, embed_dim)
        self.value_ff = nn.Linear(embed_dim, embed_dim)
        self.out_ff = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, **kwargs):
        # pytorch sparse tensors still under active development, so expect changes soon
        # for example, sparse batch matrix multiplication is not currently supported
        # TODO add support for masks
        m = query.size(0)
        n = key.size(0)
        if key.size(0) != value.size(0):
            raise RuntimeError("key and value must have same length")
        query = self.query_ff(query).view(m, -1, self.head_dim).transpose(0, 1)
        key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
        value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
        rows = torch.arange(m, device=query.device).repeat(2 * self.attn_span + 1, 1).transpose(0, 1).flatten()
        cols = torch.cat([torch.arange(i - self.attn_span, i + self.attn_span + 1, device=query.device) for i in range(n)])
        bounds = (cols >= 0) & (cols < n)
        cols[~bounds] = 0
        idxs = torch.stack([rows, cols])
        vals = (query[:, rows, :] * key[:, cols, :] * bounds.view(1, -1, 1)).sum(-1) / math.sqrt(n)
        vals[:, ~bounds] = -float("inf")
        vals = torch.dropout(torch.softmax(vals.view(-1, n, 2 * self.attn_span + 1), dim=-1), self.dropout, self.training).view(-1, idxs.size(1))
        attn_matrix = [torch.sparse.FloatTensor(idxs[:, bounds], val[bounds], (m, n)) for val in vals]
        out = self.out_ff(torch.stack([torch.sparse.mm(attn, val) for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, -1, self.embed_dim))
        return out, attn_matrix

# Use this to replace Transformer MultiheadAttention with SparseMultiheadAttention
def replace_modules(model, target, replacement, *args, **kwargs):
    for attr in dir(model):
        module = getattr(model, attr)
        if type(module) is target:
            setattr(model, attr, replacement(*args, **kwargs))
    for child in model.children():
        replace_modules(child, target, replacement, *args, **kwargs)
