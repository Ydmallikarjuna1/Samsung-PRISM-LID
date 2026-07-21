import torch

from models.ecapa_tdnn import ECAPATDNN

x = torch.randn(2, 80, 687)

model = ECAPATDNN()

x1, x2, x3 = model(x)

print("Block1 :", x1.shape)
print("Block2 :", x2.shape)
print("Block3 :", x3.shape)