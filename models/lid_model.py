import torch
import torch.nn as nn

from models.ecapa_tdnn import ECAPA_TDNN
from models.aam_softmax import AAMSoftmax


class LIDModel(nn.Module):
    """
    Complete Spoken Language Identification model.

    Pipeline:
        Log-Mel Features
              ↓
          ECAPA-TDNN
              ↓
          192-D embedding
              ↓
          AAM-Softmax
              ↓
        Language logits
    """

    def __init__(
        self,
        input_dim=80,
        channels=512,
        embedding_dim=192,
        num_classes=10,
        margin=0.2,
        scale=30.0,
    ):
        super().__init__()

        self.encoder = ECAPA_TDNN(
            input_dim=input_dim,
            channels=channels,
            embedding_dim=embedding_dim,
        )

        self.classifier = AAMSoftmax(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    def forward(self, features, labels=None):
        """
        features:
            [batch, time, n_mels]

        labels:
            [batch]

        Returns:
            If labels are provided:
                logits, loss, embeddings

            Otherwise:
                embeddings
        """

        embeddings = self.encoder(features)

        if labels is None:
            return embeddings

        logits, loss = self.classifier(
            embeddings,
            labels,
        )

        return logits, loss, embeddings


if __name__ == "__main__":
    print("Testing complete LID model...")

    batch_size = 4
    time_steps = 200
    input_dim = 80
    embedding_dim = 192
    num_classes = 10

    # Synthetic log-Mel features
    features = torch.randn(
        batch_size,
        time_steps,
        input_dim,
    )

    # Synthetic language labels
    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    model = LIDModel(
        input_dim=input_dim,
        channels=512,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    model.eval()

    with torch.no_grad():
        logits, loss, embeddings = model(
            features,
            labels,
        )

    print(f"Input shape     : {features.shape}")
    print(f"Labels shape    : {labels.shape}")
    print(f"Logits shape    : {logits.shape}")
    print(f"Embedding shape : {embeddings.shape}")
    print(f"Loss            : {loss.item():.6f}")

    assert logits.shape == (
        batch_size,
        num_classes,
    )

    assert embeddings.shape == (
        batch_size,
        embedding_dim,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(logits).all()

    print("\nCOMPLETE LID MODEL TEST PASSED")