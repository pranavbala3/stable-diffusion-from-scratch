import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class PromptEmbedding(nn.Module):
    def __init__(self, vocab_len, embed_len, token_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_len, embed_len)
        self.positional_embedding = nn.Parameter(torch.zeros(token_len, embed_len))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        
        return x 


class CLIPLayer(nn.Module):
    def __init__(self, heads, embed_len):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_len)
        self.attention = SelfAttention(heads, embed_len)
        self.layer_norm2 = nn.LayerNorm(embed_len)
        self.linear1 = nn.Linear(embed_len, 4 * embed_len)
        self.linear2 = nn.Linear(4 * embed_len, embed_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue1 = x
        
        x = self.layer_norm1(x)
        x = self.attention(x, causal=True)
        x += residue1
        
        residue2 = x
        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGeLU
        x = self.linear2(x)
        x += residue2

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PromptEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layer_norm(state)

        return output

