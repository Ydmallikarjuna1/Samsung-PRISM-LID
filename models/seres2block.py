import torch
import torch.nn as nn

from models.tdnn import TDNNBlock
from models.res2net import Res2NetBlock
from models.se import SEBlock


class SERes2Block(nn.Module):
    """
    ECAPA-TDNN SERes2 Block

    TDNN
      ↓
    Res2Net
      ↓
    TDNN
      ↓
    SE
      ↓
    Residual Add
    """

    def __init__(
        self,
        channels=512,
        kernel_size=3,
        dilation=2,
        scale=8,
        se_reduction=8,
    ):
        super().__init__()

        self.tdnn1 = TDNNBlock(
            channels,
            channels,
            kernel_size=1,
            dilation=1,
        )

        self.res2net = Res2NetBlock(
            channels=channels,
            scale=scale,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.tdnn2 = TDNNBlock(
            channels,
            channels,
            kernel_size=1,
            dilation=1,
        )

        self.se = SEBlock(
            channels=channels,
            reduction=se_reduction,
        )

    def forward(self, x):

        residual = x

        out = self.tdnn1(x)

        out = self.res2net(out)

        out = self.tdnn2(out)

        out = self.se(out)

        out = out + residual

        return out