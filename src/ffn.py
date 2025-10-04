'''
Self-Attention: mixes information from different tokens (words).

Example: "The cat sat on the mat" → “The” sees "cat sat" etc., "cat" sees "The sat on" etc.

At this stage, each token knows about other tokens, but it’s just raw, mixed info.

Feed-Forward Network (FFN): processes each token individually to turn that mixed info into something meaningful.

Example: "cat" might now encode: "subject of the sentence, singular, doing the action 'sat'"

"mat" might encode: "object, location"
'''
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (B, T, d_model)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x