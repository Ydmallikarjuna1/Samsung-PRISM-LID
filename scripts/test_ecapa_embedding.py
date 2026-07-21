import torch

from models.ecapa_tdnn import ECAPATDNN

x = torch.randn(2, 80, 687)

model = ECAPATDNN()

embedding = model(x)

print("Embedding Shape :", embedding.shape)