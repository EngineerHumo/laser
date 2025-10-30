"""Dual encoder model with attention-based global context fusion."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import GlobalEncoder, SpotEncoder


class DualEncoderMetricModel(nn.Module):
    def __init__(
        self,
        spot_embedding_dim: int = 256,
        global_feature_dim: int = 192,
        attention_dim: int = 128,
        context_dim: int = 160,
        output_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.spot_encoder = SpotEncoder(embedding_dim=spot_embedding_dim)
        self.global_encoder = GlobalEncoder(feature_dim=global_feature_dim)

        self.query_proj = nn.Linear(spot_embedding_dim, attention_dim)
        self.key_proj = nn.Linear(global_feature_dim, attention_dim)
        self.value_proj = nn.Linear(global_feature_dim, context_dim)

        self.fusion = nn.Sequential(
            nn.Linear(spot_embedding_dim + context_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, output_embedding_dim),
            nn.BatchNorm1d(output_embedding_dim),
        )

    def forward(self, spot: torch.Tensor, global_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spot_feat = self.spot_encoder(spot)
        global_feat = self.global_encoder(global_img)

        bsz, channels, height, width = global_feat.shape
        global_flat = global_feat.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

        query = self.query_proj(spot_feat).unsqueeze(1)
        keys = self.key_proj(global_flat)
        values = self.value_proj(global_flat)

        attn_logits = torch.mul(query, keys).sum(dim=-1) / math.sqrt(keys.shape[-1])
        attn_weights = torch.softmax(attn_logits, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)

        fusion_input = torch.cat([spot_feat, context], dim=-1)
        embedding = self.fusion(fusion_input)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding, spot_feat
