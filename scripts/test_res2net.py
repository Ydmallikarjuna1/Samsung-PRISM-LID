import torch

from models.res2net import Res2NetBlock

x = torch.randn(2, 512, 687)

model = Res2NetBlock(
    channels=512,
    scale=8
)

y = model(x)

print("=" * 60)
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
print("=" * 60)