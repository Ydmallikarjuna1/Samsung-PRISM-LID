import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax

    Paper:
    Margin Matters: Towards More Discriminative Deep Neural Network Embeddings
    """

    def __init__(
        self,
        embedding_dim=192,
        num_classes=3,
        margin=0.2,
        scale=30.0,
    ):
        super().__init__()

        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings, labels=None):

        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)

        cosine = F.linear(
            embeddings,
            weights,
        )

        if labels is None:
            return cosine

        sine = torch.sqrt(
            torch.clamp(
                1.0 - cosine ** 2,
                min=1e-9
            )
        )

        phi = cosine * self.cos_m - sine * self.sin_m

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1,
        )

        logits = (
            one_hot * phi +
            (1.0 - one_hot) * cosine
        )

        logits *= self.scale

        loss = F.cross_entropy(
            logits,
            labels,
        )

        return loss, logits