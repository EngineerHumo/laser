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

    batch_losses = []
    for anchor_idx in range(embeddings.size(0)):
        anchor_label = labels[anchor_idx]
        pos_mask = label_matrix[anchor_idx].clone()
        pos_mask[anchor_idx] = False
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)

        if pos_indices.numel() == 0:
            continue

        pos_dists = pairwise_dist[anchor_idx, pos_indices]
        num_pos = min(3, pos_dists.size(0))
        if pos_dists.size(0) > num_pos:
            pos_dists, _ = torch.topk(pos_dists, k=num_pos, largest=True)

        neg_mask = labels != anchor_label
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

        if neg_indices.numel() == 0:
            continue

        neg_dists = pairwise_dist[anchor_idx, neg_indices]
        num_neg = min(3, neg_dists.size(0))
        if neg_dists.size(0) > num_neg:
            neg_dists, _ = torch.topk(neg_dists, k=num_neg, largest=False)
        else:
            neg_dists = neg_dists[:num_neg]
        losses = F.relu(pos_dists.unsqueeze(1) - neg_dists.unsqueeze(0) + margin)
        batch_losses.append(losses.reshape(-1))

    if not batch_losses:
        return embeddings.new_tensor(0.0)

    return torch.cat(batch_losses).mean()
