import torch
from models.conformer.activation import GLU, Swish


class DepthWiseConvolution(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DepthWiseConvolution, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class PointWiseConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PointWiseConvolution, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConvolutionModule(torch.nn.Module):
    def __init__(self, d_model, dropout, kernel_size=3):
        super(ConvolutionModule, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            PointWiseConvolution(d_model, 2 * d_model),
            GLU(),
            DepthWiseConvolution(d_model, kernel_size, 1, int(kernel_size / 2)),
            Permute((0, 2, 1)),
            torch.nn.BatchNorm1d(d_model),
            Permute((0, 2, 1)),
            Swish(),
            PointWiseConvolution(d_model, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x)
