import torch
from models.conformer.activation import Swish


class MacaronFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        # According to Conformer paper d_ff should be 4 times of d_model
        super(MacaronFeedForward, self).__init__()
        self.feed_forward_module = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_ff),
            Swish(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.feed_forward_module(x)
