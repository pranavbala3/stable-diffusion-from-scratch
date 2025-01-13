import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, heads, model_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # these are the weight matrices for q, k, v
        self.in_project = nn.Linear(model_dim, 3 * model_dim, bias=in_proj_bias)

        # this is the weight matrix for the output
        self.out_proj = nn.Linear(model_dim, model_dim, bias=out_proj_bias)
        self.heads = heads
        self.head_dim = model_dim // heads


    def forward(self, x: torch.Tensor, causal=False) -> torch.Tensor:
        # x: (Batch_Size, seq_len, model_dim)
        batch_size, seq_len, model_dim = x.shape
        
        # (Batch_Size, seq_len, model_dim) -> (Batch_Size, seq_len, model_dim * 3) -> 3 x (Batch_Size, seq_len, model_dim)
        query, key, value = self.in_project(x).chunk(3, dim=-1)

        new_shape = (batch_size, seq_len, self.heads, self.head_dim)

        # (Batch_Size, seq_len, model_dim) -> (Batch_Size, heads, seq_len, head_dim)
        query = query.view(new_shape).transpose(1, 2)
        key = key.view(new_shape).transpose(1, 2)
        value = value.view(new_shape).transpose(1, 2)

        # self attention formula: softmax(q * k^T / dim)
        # weight: (Batch_Size, seq_len, seq_len)
        weight = query @ key.transpose(-1, -2)

        if causal:
            mask =  torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.head_dim)
        weight = F.softmax(weight, dim=-1)
 
        output = weight @ value
        output = output.transpose(1, 2)
        output = output.reshape(x.shape)
        output = self.out_proj(output)

        return output