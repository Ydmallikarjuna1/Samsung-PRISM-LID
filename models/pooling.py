import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling.

    Input:
        [batch, channels, time]

    Output:
        [batch, 2 * channels]

    The layer learns attention weights over time and
    computes weighted mean and weighted standard deviation.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(
                channels,
                attention_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(
                attention_channels,
                channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # x: [B, C, T]

        attention_logits = self.attention(x)

        # Normalize attention across time.
        attention = F.softmax(
            attention_logits,
            dim=-1,
        )

        # Weighted mean: [B, C]
        mean = torch.sum(
            attention * x,
            dim=-1,
        )

        # Weighted variance: [B, C]
        variance = torch.sum(
            attention * (x - mean.unsqueeze(-1)) ** 2,
            dim=-1,
        )

        # Numerical stability
        std = torch.sqrt(
            torch.clamp(variance, min=1e-9)
        )

        # Concatenate mean + std
        pooled = torch.cat(
            [mean, std],
            dim=1,
        )

        return pooled


if __name__ == "__main__":
    print("Testing Attentive Statistics Pooling...")

    batch_size = 2
    channels = 512
    time_steps = 100

    x = torch.randn(
        batch_size,
        channels,
        time_steps,
    )

    pooling = AttentiveStatisticsPooling(
        channels=channels,
        attention_channels=128,
    )

    y = pooling(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    assert y.shape == (
        batch_size,
        channels * 2,
    )

    assert torch.isfinite(y).all()

    print("\nATTENTIVE STATISTICS POOLING TEST PASSED")