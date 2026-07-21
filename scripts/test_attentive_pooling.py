import torch

from models.attentive_pooling import AttentiveStatisticsPooling


x = torch.randn(2, 512, 687)

model = AttentiveStatisticsPooling()

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
