import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP)

    Input:
        (B, C, T)

    Output:
        (B, 2C)
    """

    def __init__(self, channels=512, attention_channels=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),

            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        """
        x : (B, C, T)
        """

        weights = self.attention(x)

        mean = torch.sum(weights * x, dim=2)

        variance = torch.sum(
            weights * (x - mean.unsqueeze(2)) ** 2,
            dim=2
        )

        std = torch.sqrt(variance + 1e-9)

        pooled = torch.cat([mean, std], dim=1)

        return pooled