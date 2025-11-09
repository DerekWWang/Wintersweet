import timm
from timm.data import resolve_model_data_config, create_transform   

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_CHECKPOINT_CONFIG = {
    "model": "hf_hub:Snarcy/RedDino-base",
    "embed_dim": 768
}

SMALL_CHECKPOINT_CONFIG = {
    "model": "hf_hub:Snarcy/RedDino-small",
    "embed_dim": 384
}


class GatedAttentionMIL_RED(nn.Module):
    def __init__(self, checkpoint_config, M=500, L=128, attention_branches=1):
        super().__init__()
        self.M, self.L, self.B = M, L, attention_branches

        # Backbone returns a per-tile embedding (D=768 for RedDino-base, D=384 for RedDino-small)
        self.backbone = timm.create_model(checkpoint_config["model"], pretrained=True, num_classes=0)  # no classifier head
        self.embed_dim = self.backbone.num_features

        # Project to attention space
        self.feature_projector = nn.Sequential(
            nn.Linear(self.embed_dim, M),
            nn.ReLU(inplace=True),
        )

        # Gated attention
        self.attention_V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.attention_w = nn.Linear(L, self.B)

        # Bag classifier
        self.classifier = nn.Linear(M * self.B, 1)


    def forward(self, tiles, mask=None):
        """
        tiles: [K, 3, 244, 244]
        mask:  [K] boolean (optional)
        """
        H = self.backbone(tiles)          # [K, D]
        H = self.feature_projector(H)     # [K, M]

        A_V = self.attention_V(H)         # [K, L]
        A_U = self.attention_U(H)         # [K, L]
        A = self.attention_w(A_V * A_U).transpose(0, 1)  # [B, K]

        if mask is not None:
            A = A.masked_fill(~mask.unsqueeze(0), float('-inf'))

        A = F.softmax(A, dim=1)           # over tiles
        Z = A @ H                         # [B, M]
        bag_repr = Z.reshape(1, -1)
        logits = self.classifier(bag_repr)
        y_prob = torch.sigmoid(logits)  # [1, 1]
        Y_hat = (y_prob >= 0.5).float()

        return logits, y_prob, Y_hat, A
