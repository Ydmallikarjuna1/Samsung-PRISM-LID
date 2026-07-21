import torch

from models.aam_softmax import AAMSoftmax

embeddings = torch.randn(4, 192)

labels = torch.tensor([0, 1, 2, 1])

model = AAMSoftmax(
    embedding_dim=192,
    num_classes=3,
)

loss, logits = model(
    embeddings,
    labels,
)

print("Loss :", loss.item())
print("Logits Shape :", logits.shape)