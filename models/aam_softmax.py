import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax.

    Input:
        embeddings -> [batch, embedding_dim]

    Output:
        logits -> [batch, num_classes]
        loss   -> scalar
    """

    def __init__(
        self,
        embedding_dim=192,
        num_classes=10,
        margin=0.2,
        scale=30.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )

        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and class weights
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        weight = F.normalize(
            self.weight,
            p=2,
            dim=1,
        )

        # Cosine similarity
        cosine = F.linear(
            embeddings,
            weight,
        )

        # Numerical stability
        cosine = torch.clamp(
            cosine,
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )

        # Add angular margin to the correct class
        theta = torch.acos(cosine)

        target_cosine = torch.cos(
            theta + self.margin
        )

        logits = cosine.clone()

        target_mask = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).bool()

        logits[target_mask] = target_cosine[target_mask]

        # Scale logits
        logits = logits * self.scale

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits,
            labels,
        )

        return logits, loss


if __name__ == "__main__":
    print("Testing AAM-Softmax...")

    batch_size = 4
    embedding_dim = 192
    num_classes = 10

    embeddings = torch.randn(
        batch_size,
        embedding_dim,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    criterion = AAMSoftmax(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    logits, loss = criterion(
        embeddings,
        labels,
    )

    print(f"Embedding shape : {embeddings.shape}")
    print(f"Labels shape    : {labels.shape}")
    print(f"Logits shape    : {logits.shape}")
    print(f"Loss            : {loss.item():.6f}")

    assert logits.shape == (
        batch_size,
        num_classes,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("\nAAM-SOFTMAX TEST PASSED")