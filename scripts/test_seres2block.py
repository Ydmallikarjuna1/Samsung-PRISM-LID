import torch

from models.seres2block import SERes2Block


x = torch.randn(2, 512, 687)

model = SERes2Block()

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)