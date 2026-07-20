import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    Input:
        (Batch, Channels, Time)

    Output:
        (Batch, Channels, Time)
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x : (B, C, T)
        """

        b, c, _ = x.size()

        # Squeeze
        y = self.pool(x).view(b, c)

        # Excitation
        y = self.fc(y)

        # Reshape
        y = y.view(b, c, 1)

        # Scale input
        return x * y