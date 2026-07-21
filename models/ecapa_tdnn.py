import torch
import torch.nn as nn

from models.tdnn import TDNNBlock
from models.seres2block import SERes2Block
from models.attentive_pooling import AttentiveStatisticsPooling
from models.embedding import EmbeddingLayer
from models.aam_softmax import AAMSoftmax


class ECAPATDNN(nn.Module):
    """
    ECAPA-TDNN for Spoken Language Identification

    Training:
        loss, logits = model(features, labels)

    Inference:
        embeddings = model(features)
    """

    def __init__(
        self,
        input_dim=80,
        channels=512,
        embedding_dim=192,
        num_classes=107,
    ):
        super().__init__()

        # --------------------------------------------------
        # TDNN Stem
        # --------------------------------------------------
        self.stem = TDNNBlock(
        in_channels=input_dim,
        out_channels=channels,
        kernel_size=5,
        dilation=1,
)

        # --------------------------------------------------
        # SERes2 Blocks
        # --------------------------------------------------
        self.layer1 = SERes2Block(
            channels=channels,
            kernel_size=3,
            dilation=2,
        )

        self.layer2 = SERes2Block(
            channels=channels,
            kernel_size=3,
            dilation=3,
        )

        self.layer3 = SERes2Block(
            channels=channels,
            kernel_size=3,
            dilation=4,
        )

        # --------------------------------------------------
        # Multi-layer Feature Aggregation
        # --------------------------------------------------
        self.multi_layer_aggregation = nn.Sequential(
            nn.Conv1d(
                in_channels=channels * 3,
                out_channels=channels * 3,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels * 3),
        )

        # --------------------------------------------------
        # Attentive Statistics Pooling
        # --------------------------------------------------
        self.pooling = AttentiveStatisticsPooling(
            channels=channels * 3,
        )

        # --------------------------------------------------
        # Embedding Layer
        # Mean + Std = channels*3*2
        # --------------------------------------------------
        self.embedding = EmbeddingLayer(
            input_dim=channels * 3 * 2,
            embedding_dim=embedding_dim,
        )

        # --------------------------------------------------
        # AAM-Softmax Classifier
        # --------------------------------------------------
        self.classifier = AAMSoftmax(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x: (B, 80, T)
            labels: (B,) or None

        Returns:
            Training:
                loss, logits

            Inference:
                embeddings
        """

        # TDNN Stem
        x = self.stem(x)

        # SERes2 Blocks
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Multi-layer Feature Aggregation
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.multi_layer_aggregation(x)

        # Attentive Statistics Pooling
        x = self.pooling(x)

        # Embedding
        embeddings = self.embedding(x)

        # Inference
        if labels is None:
            return embeddings

        # Training
        loss, logits = self.classifier(
            embeddings,
            labels,
        )

        return loss, logits