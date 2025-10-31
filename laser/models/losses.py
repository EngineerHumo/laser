"""Loss functions used for deep metric learning training."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    """Implements ArcFace margin-based classification head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = self._cosine(input)
        cosine_sq = torch.clamp(cosine ** 2, 0.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine_sq, min=1e-6))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        #print(label)
        #print(self.out_features)
        one_hot = F.one_hot(label, num_classes=self.out_features).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits

    def inference(self, input: torch.Tensor) -> torch.Tensor:
        return self._cosine(input) * self.s

    def _cosine(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = F.normalize(input)
        normalized_weight = F.normalize(self.weight)
        return F.linear(normalized_input, normalized_weight)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Compute batch-hard triplet loss on L2-normalised embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D tensor of shape (batch, dim)")

    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = label_matrix ^ torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    mask_neg = ~label_matrix

    pos_dist = pairwise_dist.clone()
    pos_dist[~mask_pos] = 0.0
    hardest_pos = pos_dist.max(dim=1)[0]

    neg_dist = pairwise_dist.clone()
    neg_dist[~mask_neg] = float("inf")
    hardest_neg = neg_dist.min(dim=1)[0]
    hardest_neg[~torch.isfinite(hardest_neg)] = 0.0

    losses = F.relu(hardest_pos - hardest_neg + margin)
    return losses.mean()
