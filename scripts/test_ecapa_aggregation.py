import torch

from models.ecapa_tdnn import ECAPATDNN

x = torch.randn(2, 80, 687)

model = ECAPATDNN()

y = model(x)

print("Output Shape :", y.shape)