import torch

from models.embedding import EmbeddingLayer

x = torch.randn(2, 1024)

model = EmbeddingLayer()

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)