import torch

from models.ecapa_tdnn import ECAPATDNN

model = ECAPATDNN(num_classes=107)

features = torch.randn(4, 80, 687)

labels = torch.randint(0, 107, (4,))

loss, logits = model(features, labels)

print("Loss :", loss.item())
print("Logits Shape :", logits.shape)