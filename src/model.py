'''
Now the entire Transformer Based Language Model is written.

The language model contains two embeddings:
    - Token Embedding => This is the relationship between tokens (vocab_size, d_model)
    - Positional Embedding => This is embedding is used to know the position of the token in the sentence (seq_len, d_model)
pass them both to dropout after adding both the embedding

Now the transformer blocks can be multiplied as many layers as required

Finally the output of the transformer block is passed through layer normalization

A linear layer is required to map embeddings to vocab_size or our logits by which we predict our next token.
'''

import torch
import torch.nn as nn
from src.transformer import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(self, tokenizer, d_model, seq_length, d_ff, n_layers, n_heads, ffn_factor=4, dropout=0.1):
        super().__init__()
        vocab_size = tokenizer.vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.trf_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff*ffn_factor, n_heads) for _ in range(n_layers)
        ]
        )

        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.seq_len = seq_length
        self._init_weights()

    def _init_weights(self):
        for parameters in self.parameters():
            if parameters.dim() > 1:
                nn.init.xavier_uniform_(parameters)


    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape

        # creating position indices
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0) # (1, T)

        # embeddings
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        for blocks in self.trf_blocks:
            x = blocks(x, attn_mask=attn_mask)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits
    