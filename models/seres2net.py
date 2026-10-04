import torch
import torch.nn as nn

from models.res2net import Res2NetBlock
from models.se import SEBlock


class SERes2NetBlock(nn.Module):
    """
    Squeeze-and-Excitation Res2Net block.

    Combines:
        Res2Net multi-scale feature extraction
        +
        Squeeze-and-Excitation channel attention

    Input:
        [batch, channels, time]

    Output:
        [batch, out_channels, time]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale=8,
        kernel_size=3,
        dilation=2,
        se_reduction=8,
    ):
        super().__init__()

        self.res2net = Res2NetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.se = SEBlock(
            channels=out_channels,
            reduction=se_reduction,
        )

    def forward(self, x):
        x = self.res2net(x)
        x = self.se(x)

        return x


if __name__ == "__main__":
    print("Testing SERes2Net block...")

    batch_size = 2
    input_channels = 512
    output_channels = 512
    time_steps = 100

    x = torch.randn(
        batch_size,
        input_channels,
        time_steps,
    )

    block = SERes2NetBlock(
        in_channels=input_channels,
        out_channels=output_channels,
        scale=8,
        kernel_size=3,
        dilation=2,
        se_reduction=8,
    )

    y = block(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == (
        batch_size,
        output_channels,
        time_steps,
    )

    assert torch.isfinite(y).all()

    print("\nSERES2NET BLOCK TEST PASSED")