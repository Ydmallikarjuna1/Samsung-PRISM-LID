import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Projects the pooled ECAPA representation
    into a compact embedding space.

    Input:
        [batch, input_dim]

    Output:
        [batch, embedding_dim]
    """

    def __init__(
        self,
        input_dim=1024,
        embedding_dim=192,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            embedding_dim,
        )

        self.batch_norm = nn.BatchNorm1d(
            embedding_dim
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)

        return x


if __name__ == "__main__":
    print("Testing embedding layer...")

    batch_size = 2
    input_dim = 1024
    embedding_dim = 192

    x = torch.randn(
        batch_size,
        input_dim,
    )

    embedding = EmbeddingLayer(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
    )

    y = embedding(x)

    print(f"Input shape     : {x.shape}")
    print(f"Embedding shape : {y.shape}")

    assert y.shape == (
        batch_size,
        embedding_dim,
    )

    assert torch.isfinite(y).all()

    print("\nEMBEDDING LAYER TEST PASSED")