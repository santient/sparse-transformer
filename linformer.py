class MultiheadLinearAttention(nn.Module):
    """Based on "Linformer: Self-Attention with Linear Complexity" (https://arxiv.org/abs/2006.04768)"""
    def __init__(self, embed_dim, project_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = 1 / math.sqrt(self.head_dim)
        self.query_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.key_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.value_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.key_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.value_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        tgt_len = query.size(0)
        src_len = key.size(0)
        bs = query.size(1)
        q = self.query_embed_linear(query).view(tgt_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.key_embed_linear(key).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.value_embed_linear(value).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        e = self.key_project_linear(key).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        f = self.value_project_linear(value).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        attn = self.scale * q @ (e @ k).transpose(1, 2)
        # masking code from PyTorch MultiheadAttention source code
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = attn.view(bs * self.num_heads, tgt_len, self.project_dim)
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout, training=self.training)
        out = attn @ (f @ v)
        out = self.out_linear(out.transpose(0, 1).contiguous().view(tgt_len, bs, self.embed_dim))
        if need_weights:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim).sum(dim=1) / self.num_heads
            return out, attn
        else:
            return out, None