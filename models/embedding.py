import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        input_dim=3072,
        embedding_dim=192,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        return self.embedding(x)