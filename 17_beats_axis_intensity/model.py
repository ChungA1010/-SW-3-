"""BEATs-based single-axis intensity model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from beats_encoder import BEATs, BEATsConfig


def build_default_beats_config() -> BEATsConfig:
    cfg = BEATsConfig()
    cfg.input_patch_size = 16
    cfg.embed_dim = 512
    cfg.encoder_layers = 12
    cfg.encoder_embed_dim = 768
    cfg.encoder_ffn_embed_dim = 3072
    cfg.encoder_attention_heads = 12
    cfg.finetuned_model = False
    cfg.predictor_class = 527
    return cfg


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        logits = self.score(hidden_states).squeeze(-1)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return torch.sum(hidden_states * weights, dim=1)


class BEATsAxisIntensity(nn.Module):
    """Predict one axis intensity: off/50/100."""

    def __init__(
        self,
        axis_name: str,
        pretrained_checkpoint_path: str | Path | None = None,
        backbone_config_override: dict | None = None,
    ) -> None:
        super().__init__()
        self.axis_name = axis_name
        checkpoint_path = Path(pretrained_checkpoint_path).expanduser().resolve() if pretrained_checkpoint_path else None
        checkpoint = None
        if backbone_config_override is not None:
            cfg = BEATsConfig(backbone_config_override)
            cfg.finetuned_model = False
        elif checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            cfg = BEATsConfig(checkpoint["cfg"])
            cfg.finetuned_model = False
        else:
            cfg = build_default_beats_config()

        self.backbone = BEATs(cfg)
        self.backbone_config = dict(cfg.__dict__)
        self.pretrained_checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.pretrained_loaded = False
        if checkpoint is not None:
            self.backbone.load_state_dict(checkpoint["model"], strict=True)
            self.pretrained_loaded = True

        hidden_size = cfg.encoder_embed_dim
        self.pool = MaskedAttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, 3),
        )

    def forward(self, waveforms: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states, output_padding_mask = self.backbone.extract_features(
            waveforms,
            padding_mask=padding_mask,
        )
        pooled = self.pool(hidden_states, output_padding_mask)
        return self.head(pooled)
