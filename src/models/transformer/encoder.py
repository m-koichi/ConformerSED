import torch
from models.transformer.attention import MultiHeadedAttention
from models.transformer.embedding import PositionalEncoding
from models.transformer.encoder_layer import EncoderLayer
from models.transformer.layer_norm import LayerNorm
from models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from models.transformer.repeat import repeat
from models.transformer.subsampling import Conv2dNoSubsampling, Conv2dSubsampling

# Reference: https://github.com/espnet/espnet/tree/master/espnet/nets/pytorch_backend/transformer


class Encoder(torch.nn.Module):
    """Encoder module
    :param int idim: input dim
    :param argparse.Namespace args: experiment config
    """

    def __init__(self, idim, args, pos_enc=True):
        super(Encoder, self).__init__()
        if args.transformer_input_layer == "linear":
            if pos_enc:
                self.input_layer = torch.nn.Sequential(
                    torch.nn.Linear(idim, args.adim),
                    torch.nn.LayerNorm(args.adim),
                    torch.nn.Dropout(args.dropout_rate),
                    torch.nn.ReLU(),
                    PositionalEncoding(args.adim, args.dropout_rate),
                )
            else:
                self.input_layer = torch.nn.Sequential(
                    torch.nn.Linear(idim, args.adim),
                    torch.nn.LayerNorm(args.adim),
                    torch.nn.Dropout(args.dropout_rate),
                    torch.nn.ReLU(),
                )

        elif args.transformer_input_layer == "conv2d":
            self.input_layer = Conv2dSubsampling(idim, args.adim, args.dropout_rate)
        elif args.transformer_input_layer == "conv2d_no":
            self.input_layer = Conv2dNoSubsampling(idim, args.adim, args.dropout_rate)
        elif args.transformer_input_layer == "embed":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Embedding(idim, args.adim), PositionalEncoding(args.adim, args.dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + args.transformer_input_layer)

        self.encoders = repeat(
            args.elayers,
            lambda: EncoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.transformer_attn_dropout_rate),
                PositionwiseFeedForward(args.adim, args.eunits, args.dropout_rate),
                args.dropout_rate,
                args.after_conv,
            ),
        )
        self.norm = LayerNorm(args.adim)

    def forward(self, x, mask=None):
        """Embed positions in tensor
        :param torch.Tensor x: input tensor
        :param torch.Tensor mask: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.input_layer, Conv2dNoSubsampling):
            x, mask = self.input_layer(x, mask)
        elif isinstance(self.input_layer, Conv2dSubsampling):
            x, mask = self.input_layer(x, mask)
        else:
            x = self.input_layer(x)
        #         x, mask = self.encoders(x, mask)
        #         return x, mask
        x, mask = self.encoders(x, mask)
        return self.norm(x), mask
