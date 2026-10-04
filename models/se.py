import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Input:
        [batch, channels, time]

    Output:
        [batch, channels, time]
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        if channels % reduction != 0:
            raise ValueError(
                "channels must be divisible by reduction"
            )

        hidden_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Squeeze: [B, C, T] -> [B, C]
        scale = self.pool(x).squeeze(-1)

        # Excitation: [B, C] -> [B, C]
        scale = self.fc(scale)

        # Restore time dimension: [B, C] -> [B, C, 1]
        scale = scale.unsqueeze(-1)

        # Re-weight channels
        return x * scale


if __name__ == "__main__":
    print("Testing SE block...")

    batch_size = 2
    channels = 512
    time_steps = 100

    x = torch.randn(
        batch_size,
        channels,
        time_steps,
    )

    block = SEBlock(
        channels=channels,
        reduction=8,
    )

    y = block(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    print("\nSE BLOCK TEST PASSED")