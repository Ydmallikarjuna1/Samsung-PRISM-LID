import torch

from models.se import SEBlock


x = torch.randn(2, 512, 687)

model = SEBlock(
    channels=512,
    reduction=8
)

y = model(x)

print("=" * 60)
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
print("=" * 60)