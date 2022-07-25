from W4DTS.core.ME_unet_squantial import MinkUNetDecoder_fe as MinkUNetDecoder, MinkUNetEncoder
from torch import nn
import torch
from MinkowskiEngine.modules.resnet_block import BasicBlock
import MinkowskiEngine as ME

class Model(nn.Module):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    SCALE = 16

    def __init__(self, in_channels, out_channels,D = 3):
        nn.Module.__init__(self)
        self.main_encoder = MinkUNetEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            D=3,
            BLOCK=self.BLOCK,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
        )
        self.decoder = MinkUNetDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            D=3,
            BLOCK=self.BLOCK,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
        )

    def forward(
            self, x
    ):
        res_m = list(self.main_encoder(x))
        out, res_f = self.decoder(res_m)
        return out, res_f#res_m[-1],
