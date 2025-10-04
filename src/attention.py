'''
B -> Batch Size,
T -> Seq Length
C or d_model -> Embedding Dimension for each token
'''

import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = d_model//n_heads

        self.qkv = nn.Linear(d_model, 3*d_model) # This is for Q, K, V
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.5)
        self.proj_dropout = nn.Dropout(0.5)

    def forward(self, x, attn_mask=None):
        # x (input) -> (B, T, d_model)
        B, T, C = x.shape

        qkv = self.qkv(x) # qkv -> (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1) # Now each q, k, v -> (B, T, d_model)
        q = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # now q -> (B, n_heads, T, head_dim) and we can do attention per head easily
        k = k.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # similarly for k
        v = v.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # similarly for v
        # here the view breaks down the d_model into n_heads and head_dim making sure each attention vectors have multiple heads

        #calculate attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)# scores -> (B, n_heads, T, T)
        if attn_mask is not None:
            scores = scores.mask_fill(~attn_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1) # why the last column, because (B, n_heads, query T, key T) and we want to see probability distribution across key for each query
        attn = self.attn_dropout(attn)
        
        out = attn @ v # out -> (B, n_heads, T, head_dim)

        # Merging heads
        out = out.transpose(1, 2).contiguous().view(B, T, C) # out -> (B, T, d_model)
        out = self.out(out) # out -> (B, T, d_model)
        out = self.proj_dropout(out)
        return out
        


