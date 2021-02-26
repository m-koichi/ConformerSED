import torch
from models.conformer.attention import RelMultiHeadAttn
from models.conformer.convolution import ConvolutionModule
from models.conformer.macaron_feed_forward import MacaronFeedForward


class ConformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout, kernel_size):
        super(ConformerBlock, self).__init__()
        self.ffn1 = MacaronFeedForward(d_model, d_ff, dropout)
        self.mhsa = RelMultiHeadAttn(n_head, d_model, dropout)
        self.conv = ConvolutionModule(d_model, dropout, kernel_size)
        self.ffn2 = MacaronFeedForward(d_model, d_ff, dropout)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = 0.5 * self.ffn1(x) + x
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.mhsa(x, mask)  # (T, B, D)
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.conv(x) + x
        x = 0.5 * self.ffn2(x) + x
        x = self.norm(x)
        return x, mask
