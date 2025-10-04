'''
Transformer Block is basically combination of:
    LayerNorm → normalizes token embeddings for stability
    Multi-Head Self-Attention (MHSA) → lets tokens attend to other tokens
    Residual connection → adds input back to output (helps gradients flow)
    LayerNorm → again, before FFN
    Feed-Forward Network (FFN) → position-wise transformation
    Residual connection → add input of FFN back
'''

import torch.nn as nn
from src.attention import MultiHeadSelfAttention
from src.ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff, dropout=0.5)

    def forward(self, x, attn_mask=None):
        '''
        x: (B, T, d_model)
        '''
        # MHSA + residual
        _x = self.ln1(x)
        x = x + self.attn(_x, attn_mask=attn_mask)

        # MHSA + residual
        _x = self.ln2(x)
        x = x + self.ffn(_x)

        return x