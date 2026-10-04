import torch
import torch.nn as nn


class Res2NetBlock(nn.Module):
    """
    Res2Net-style multi-scale feature block.

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
        dilation=1,
    ):
        super().__init__()

        if out_channels % scale != 0:
            raise ValueError(
                "out_channels must be divisible by scale"
            )

        self.scale = scale
        self.width = out_channels // scale

        self.input_projection = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=((kernel_size - 1) // 2) * dilation,
                )
                for _ in range(scale - 1)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(self.width)
                for _ in range(scale - 1)
            ]
        )

        self.activation = nn.ReLU()

        self.output_projection = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
        )

        self.output_norm = nn.BatchNorm1d(
            out_channels
        )

    def forward(self, x):
        residual = self.input_projection(x)

        # Split channels into multiple scale groups.
        chunks = torch.chunk(
            residual,
            self.scale,
            dim=1,
        )

        outputs = [chunks[0]]

        for i in range(1, self.scale):
            if i == 1:
                current = chunks[i]
            else:
                current = chunks[i] + outputs[-1]

            current = self.convs[i - 1](current)
            current = self.batch_norms[i - 1](current)
            current = self.activation(current)

            outputs.append(current)

        out = torch.cat(outputs, dim=1)

        out = self.output_projection(out)
        out = self.output_norm(out)

        out = out + residual
        out = self.activation(out)

        return out


if __name__ == "__main__":
    print("Testing Res2Net block...")

    batch_size = 2
    input_channels = 512
    time_steps = 100

    x = torch.randn(
        batch_size,
        input_channels,
        time_steps,
    )

    block = Res2NetBlock(
        in_channels=512,
        out_channels=512,
        scale=8,
        kernel_size=3,
        dilation=2,
    )

    y = block(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == (2, 512, 100)
    assert torch.isfinite(y).all()

    print("\nRES2NET BLOCK TEST PASSED")