import torch
import torch.nn as nn


class Res2NetBlock(nn.Module):
    """
    Res2Net Block used in ECAPA-TDNN
    """

    def __init__(
        self,
        channels,
        scale=8,
        kernel_size=3,
        dilation=2,
    ):
        super().__init__()

        assert channels % scale == 0, \
            "channels must be divisible by scale"

        self.scale = scale
        self.width = channels // scale

        self.blocks = nn.ModuleList()

        for _ in range(scale - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.width,
                        self.width,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=((kernel_size - 1) // 2) * dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.width),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """
        x : [B, C, T]
        """

        splits = torch.split(x, self.width, dim=1)

        outputs = [splits[0]]

        for i in range(1, self.scale):
            if i == 1:
                out = self.blocks[i - 1](splits[i])
            else:
                out = self.blocks[i - 1](splits[i] + outputs[-1])

            outputs.append(out)

        return torch.cat(outputs, dim=1)