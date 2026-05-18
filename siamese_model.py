"""
siamese_model.py
----------------
PyTorch 기반 샴 네트워크 (Siamese Network).

두 음원 피처 벡터를 입력받아
'얼마나 가까운지(유사도)'를 end-to-end로 학습.

손실 함수: Contrastive Loss
  - 유사한 쌍(similar pair): 거리 최소화
  - 다른 쌍(dissimilar pair): 마진(margin)만큼 거리 확보
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────
# 공유 인코더 네트워크
# ──────────────────────────────────────────

class ToneEncoder(nn.Module):
    """
    오디오 피처 벡터 → 저차원 임베딩 공간으로 매핑.
    두 브랜치(원본, 유저)가 이 네트워크의 가중치를 공유.

    구조:
        FC(input_dim → 512) → BN → ReLU → Dropout
        FC(512 → 256)        → BN → ReLU → Dropout
        FC(256 → embedding_dim)
    """

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ──────────────────────────────────────────
# 샴 네트워크
# ──────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    두 피처 벡터를 받아 임베딩 공간에서의 유클리드 거리를 출력.
    거리 → 유사도 변환: similarity = 1 / (1 + distance)
    """

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        # 두 브랜치가 동일한 인코더를 공유 (가중치 공유)
        self.encoder = ToneEncoder(input_dim, embedding_dim)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            emb1      : 첫 번째 임베딩
            emb2      : 두 번째 임베딩
            distance  : L2 거리
        """
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        distance = F.pairwise_distance(emb1, emb2)
        return emb1, emb2, distance

    def get_similarity(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> float:
        """
        0~1 범위의 유사도 점수 반환.
        distance=0 → similarity=1.0 (완전 일치)
        """
        with torch.no_grad():
            _, _, dist = self.forward(x1, x2)
            similarity = 1.0 / (1.0 + dist.item())
        return similarity


# ──────────────────────────────────────────
# Contrastive Loss
# ──────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    유사한 쌍의 거리는 좁히고,
    다른 쌍의 거리는 마진 이상으로 벌린다.

    L = (1-y) * D^2 + y * max(0, margin - D)^2
      y=0: similar pair, y=1: dissimilar pair
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,
        label: torch.Tensor,     # 0=유사, 1=다름
    ) -> torch.Tensor:
        loss_similar    = (1 - label) * distance.pow(2)
        loss_dissimilar = label * F.relu(self.margin - distance).pow(2)
        return torch.mean(loss_similar + loss_dissimilar)


# ──────────────────────────────────────────
# 학습 루프 (미니 예시)
# ──────────────────────────────────────────

def train_siamese(
    model: SiameseNetwork,
    train_pairs: list[tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SiameseNetwork:
    """
    샴 네트워크 학습.

    Args:
        train_pairs : [(feat_ref, feat_user, label), ...]
                      label: 0=유사한 쌍, 1=다른 쌍
        epochs      : 학습 에포크 수
        lr          : 학습률

    Returns:
        학습된 SiameseNetwork
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for feat1, feat2, label in train_pairs:
            x1 = torch.tensor(feat1, dtype=torch.float32).unsqueeze(0).to(device)
            x2 = torch.tensor(feat2, dtype=torch.float32).unsqueeze(0).to(device)
            y  = torch.tensor([label], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            _, _, dist = model(x1, x2)
            loss = criterion(dist, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_pairs)
            print(f"[Siamese] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return model
