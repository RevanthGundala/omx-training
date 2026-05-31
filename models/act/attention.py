import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=2048, dropout=0.1, use_cross_attention=False):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(dim, num_heads, dropout)
        self.cross_attn = Attention(dim, num_heads, dropout, use_cross_attention=True) if use_cross_attention else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim) if use_cross_attention else None
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_out=None, future_mask=None):
        attn_output = self.self_attn(self.self_norm(x), future_mask=future_mask)
        x = x + attn_output

        if self.cross_attn is not None and encoder_out is not None:
            attn_output = self.cross_attn(self.cross_norm(x), encoder_out=encoder_out)
            x = x + attn_output

        mlp_output = self.mlp(self.mlp_norm(x))
        return x + mlp_output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_cross_attention=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_q_proj = nn.Linear(dim, dim)
            self.cross_kv_proj = nn.Linear(dim, dim * 2)

    def forward(self, x, encoder_out=None, future_mask=None):
        B, T, C = x.size()
        if self.use_cross_attention and encoder_out is not None:
            S = encoder_out.size(1)
            cross_q = self.cross_q_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            cross_kv = self.cross_kv_proj(encoder_out).reshape(B, S, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            cross_k, cross_v = cross_kv[0], cross_kv[1]
            attn_weights = (cross_q @ cross_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if future_mask is not None:
                attn_weights = attn_weights.masked_fill(future_mask == 0, torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = (attn_weights @ cross_v).transpose(1, 2).reshape(B, T, C)
            return self.out_proj(attn_output)

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if future_mask is not None:
            attn_weights = attn_weights.masked_fill(future_mask == 0, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)


