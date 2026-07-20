import torch

from models.tdnn import TDNNBlock

# Dummy FBank feature
# Batch = 2
# Channels = 80 Mel bins
# Frames = 687

x = torch.randn(2, 80, 687)

model = TDNNBlock(
    in_channels=80,
    out_channels=512,
    kernel_size=5,
    dilation=1
)

y = model(x)

print("=" * 60)
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
print("=" * 60)