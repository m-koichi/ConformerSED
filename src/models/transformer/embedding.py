import math

import torch

# Reference: https://github.com/espnet/espnet/tree/master/espnet/nets/pytorch_backend/transformer


class PositionalEncoding(torch.nn.Module):
    """Positional encoding module
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.max_len = max_len
        self.xscale = math.sqrt(d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)
