import torch
import torch.nn as nn

from models.tdnn import TDNNBlock
from models.seres2net import SERes2NetBlock
from models.pooling import AttentiveStatisticsPooling
from models.embedding import EmbeddingLayer


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN baseline for spoken language identification.

    Input:
        Log-Mel features: [batch, time, n_mels]

    Output:
        Language embeddings: [batch, embedding_dim]
    """

    def __init__(
        self,
        input_dim=80,
        channels=512,
        embedding_dim=192,
    ):
        super().__init__()

        self.tdnn = TDNNBlock(
            in_channels=input_dim,
            out_channels=channels,
            kernel_size=5,
            dilation=1,
        )

        self.seres2net1 = SERes2NetBlock(
            in_channels=channels,
            out_channels=channels,
            scale=8,
            kernel_size=3,
            dilation=2,
            se_reduction=8,
        )

        self.seres2net2 = SERes2NetBlock(
            in_channels=channels,
            out_channels=channels,
            scale=8,
            kernel_size=3,
            dilation=3,
            se_reduction=8,
        )

        self.seres2net3 = SERes2NetBlock(
            in_channels=channels,
            out_channels=channels,
            scale=8,
            kernel_size=3,
            dilation=4,
            se_reduction=8,
        )

        # Concatenation of the outputs from the three
        # SE-Res2Net blocks.
        self.mfa = nn.Conv1d(
            channels * 3,
            channels,
            kernel_size=1,
        )

        self.mfa_norm = nn.BatchNorm1d(
            channels
        )

        self.mfa_activation = nn.ReLU()

        self.pooling = AttentiveStatisticsPooling(
            channels=channels,
            attention_channels=128,
        )

        self.embedding = EmbeddingLayer(
            input_dim=channels * 2,
            embedding_dim=embedding_dim,
        )

    def forward(self, features):
        """
        features:
            [batch, time, n_mels]

        returns:
            [batch, embedding_dim]
        """

        if features.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, time, n_mels], "
                f"got {features.shape}"
            )

        # Convert:
        # [batch, time, features]
        # ->
        # [batch, features, time]
        x = features.transpose(1, 2)

        x = self.tdnn(x)

        x1 = self.seres2net1(x)
        x2 = self.seres2net2(x1)
        x3 = self.seres2net3(x2)

        # Multi-layer feature aggregation
        x = torch.cat(
            [x1, x2, x3],
            dim=1,
        )

        x = self.mfa(x)
        x = self.mfa_norm(x)
        x = self.mfa_activation(x)

        # Attentive mean + standard deviation
        x = self.pooling(x)

        # Final language embedding
        embedding = self.embedding(x)

        return embedding


if __name__ == "__main__":
    print("Testing complete ECAPA-TDNN...")

    batch_size = 2
    time_steps = 200
    input_dim = 80
    embedding_dim = 192

    features = torch.randn(
        batch_size,
        time_steps,
        input_dim,
    )

    model = ECAPA_TDNN(
        input_dim=input_dim,
        channels=512,
        embedding_dim=embedding_dim,
    )

    model.eval()

    with torch.no_grad():
        embeddings = model(features)

    print(f"Input shape     : {features.shape}")
    print(f"Embedding shape : {embeddings.shape}")

    assert embeddings.shape == (
        batch_size,
        embedding_dim,
    )

    assert torch.isfinite(embeddings).all()

    print("\nECAPA-TDNN MODEL TEST PASSED")