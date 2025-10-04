'''
We also need to cut the long sequence of token IDs into training chunks of length seq_len (from config).

Each training example looks like:
input: "The cat sat on the"
target: "he cat sat on the mat" (shifted by 1 token).
'''

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len, pad_id):
        self.tokens = token_ids
        self.seq_len = seq_len
        self.pad_id = pad_id
    
    def __len__(self):
        return max(1, len(self.tokens))

    def __getitem__(self, idx):
        x = self.tokens[idx : idx+self.seq_len]
        y = self.tokens[idx+1 : idx+self.seq_len+1]

        x += [self.pad_id] * (self.seq_len - len(x))
        y += [self.pad_id] * (self.seq_len - len(y))

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
