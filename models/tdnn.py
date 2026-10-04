import torch
import torch.nn as nn


class TDNNBlock(nn.Module):
    """
    Time-Delay Neural Network block.

    Input:
        [batch, channels, time]

    Output:
        [batch, out_channels, time]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        dilation=1,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()

        self.batch_norm = nn.BatchNorm1d(
            out_channels
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)

        return x


if __name__ == "__main__":
    print("Testing TDNN block...")

    batch_size = 2
    input_channels = 80
    time_steps = 100

    x = torch.randn(
        batch_size,
        input_channels,
        time_steps,
    )

    block = TDNNBlock(
        in_channels=80,
        out_channels=512,
        kernel_size=5,
        dilation=1,
    )

    y = block(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == (2, 512, 100)
    assert torch.isfinite(y).all()

    print("\nTDNN BLOCK TEST PASSED")