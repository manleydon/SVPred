"""Small feature-level models for privileged street-view distillation."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProjectionBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        middle_dim = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SatelliteStudent(nn.Module):
    """Satellite-only model used for both the supervised and KD students."""

    def __init__(
        self,
        satellite_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(self, satellite: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.projector(satellite)
        prediction = self.head(embedding)
        return prediction, embedding


class MultimodalTeacher(nn.Module):
    """Gated residual fusion teacher using real street view at training/test Oracle time."""

    def __init__(
        self,
        satellite_dim: int,
        street_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.street_projector = ProjectionBlock(street_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(
        self,
        satellite: torch.Tensor,
        street: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        satellite_embedding = self.satellite_projector(satellite)
        street_embedding = self.street_projector(street)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused_embedding = self.fusion_norm(
            satellite_embedding + gate * street_embedding
        )
        prediction = self.head(fused_embedding)
        return prediction, fused_embedding, gate


class StreetFeatureHallucinator(nn.Module):
    """Simple satellite-to-street feature regression baseline."""

    def __init__(
        self,
        satellite_dim: int,
        street_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(satellite_dim),
            nn.Linear(satellite_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, street_dim),
        )

    def forward(self, satellite: torch.Tensor) -> torch.Tensor:
        return self.net(satellite)


class TaskRelevantBottleneckTeacher(nn.Module):
    """Real-SV teacher with a compact, reconstructive street bottleneck.

    The reconstruction head prevents the supervised bottleneck from becoming an
    unconstrained label code.  It is used only while fitting the privileged
    teacher; downstream inference never requires street view.
    """

    def __init__(
        self,
        satellite_dim: int,
        street_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.street_encoder = nn.Sequential(
            nn.LayerNorm(street_dim),
            nn.Linear(street_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.street_lift = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)
        self.street_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, street_dim),
        )

    def encode_street(self, street: torch.Tensor) -> torch.Tensor:
        return self.street_encoder(street)

    def forward(
        self,
        satellite: torch.Tensor,
        street: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        satellite_embedding = self.satellite_projector(satellite)
        bottleneck = self.encode_street(street)
        street_embedding = self.street_lift(bottleneck)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused_embedding = self.fusion_norm(
            satellite_embedding + gate * street_embedding
        )
        prediction = self.head(fused_embedding)
        reconstruction = self.street_decoder(bottleneck)
        return prediction, bottleneck, gate, reconstruction


class QualityAwareStreetSetBottleneckTeacher(nn.Module):
    """Task bottleneck teacher that softly rejects poor views in a street set.

    Quality is inferred only from the street-view set, never from the target or
    satellite input.  This makes the resulting bottleneck a clean privileged
    street representation for a satellite-only student to distil.
    """

    def __init__(
        self,
        satellite_dim: int,
        street_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.view_encoder = nn.Sequential(
            nn.LayerNorm(street_dim),
            nn.Linear(street_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        # The score uses a view and its set context.  Kept deliberately small
        # for N=2300 and to prevent selection from becoming a hidden predictor.
        self.quality_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.street_encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.street_lift = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)
        self.street_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, street_dim),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(values.dtype)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        satellite: torch.Tensor,
        street_set: torch.Tensor,
        street_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if street_set.ndim != 3 or street_mask.ndim != 2:
            raise ValueError("Expected street_set [B,K,D] and street_mask [B,K].")
        if not torch.all(street_mask.any(dim=1)):
            raise ValueError("Every sample must retain at least one street view.")
        encoded_views = self.view_encoder(street_set)
        set_mean = self._masked_mean(encoded_views, street_mask)
        context = set_mean.unsqueeze(1).expand_as(encoded_views)
        logits = self.quality_head(torch.cat([encoded_views, context], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~street_mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)
        pooled = torch.sum(encoded_views * attention.unsqueeze(-1), dim=1)
        bottleneck = self.street_encoder(pooled)
        satellite_embedding = self.satellite_projector(satellite)
        street_embedding = self.street_lift(bottleneck)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused = self.fusion_norm(satellite_embedding + gate * street_embedding)
        prediction = self.head(fused)
        reconstruction = self.street_decoder(bottleneck)
        return prediction, bottleneck, gate, reconstruction, attention


class SatelliteBottleneckStudent(nn.Module):
    """Satellite-only model that hallucinates a privileged street bottleneck."""

    def __init__(
        self,
        satellite_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.bottleneck_predictor = nn.Sequential(
            nn.LayerNorm(satellite_dim),
            nn.Linear(satellite_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.street_lift = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(
        self,
        satellite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        satellite_embedding = self.satellite_projector(satellite)
        predicted_bottleneck = self.bottleneck_predictor(satellite)
        street_embedding = self.street_lift(predicted_bottleneck)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused_embedding = self.fusion_norm(
            satellite_embedding + gate * street_embedding
        )
        prediction = self.head(fused_embedding)
        return prediction, predicted_bottleneck, gate


class SharedPrivateBottleneckTeacher(nn.Module):
    """Decompose a compact street representation into shared and private parts.

    The full privileged prediction adds both street contributions to the
    satellite embedding.  ``satellite_shared`` is used only to make the street
    shared component structurally predictable from satellite imagery.
    """

    def __init__(
        self,
        satellite_dim: int,
        street_dim: int,
        shared_dim: int = 32,
        private_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.street_trunk = ProjectionBlock(street_dim, hidden_dim, dropout)
        self.street_shared_head = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim), nn.LayerNorm(shared_dim)
        )
        self.street_private_head = nn.Sequential(
            nn.Linear(hidden_dim, private_dim), nn.LayerNorm(private_dim)
        )
        self.satellite_shared_head = nn.Sequential(
            nn.LayerNorm(satellite_dim),
            nn.Linear(satellite_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        self.shared_lift = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.private_lift = nn.Sequential(
            nn.Linear(private_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.shared_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.private_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)
        self.street_decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, street_dim),
        )

    def encode_street(
        self, street: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.street_trunk(street)
        return self.street_shared_head(trunk), self.street_private_head(trunk)

    def forward(
        self,
        satellite: torch.Tensor,
        street: torch.Tensor,
        include_private: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        satellite_embedding = self.satellite_projector(satellite)
        street_shared, street_private = self.encode_street(street)
        satellite_shared = self.satellite_shared_head(satellite)
        shared_embedding = self.shared_lift(street_shared)
        private_embedding = self.private_lift(street_private)
        shared_gate = self.shared_gate(
            torch.cat([satellite_embedding, shared_embedding], dim=-1)
        )
        private_gate = self.private_gate(
            torch.cat([satellite_embedding, private_embedding], dim=-1)
        )
        fused = satellite_embedding + shared_gate * shared_embedding
        if include_private:
            fused = fused + private_gate * private_embedding
        prediction = self.head(self.fusion_norm(fused))
        reconstruction = self.street_decoder(
            torch.cat([street_shared, street_private], dim=-1)
        )
        return (
            prediction,
            street_shared,
            street_private,
            satellite_shared,
            shared_gate,
            private_gate,
            reconstruction,
        )


class SharedBottleneckStudent(nn.Module):
    """Satellite-only student that predicts only the recoverable shared part."""

    def __init__(
        self,
        satellite_dim: int,
        shared_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.shared_predictor = nn.Sequential(
            nn.LayerNorm(satellite_dim),
            nn.Linear(satellite_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.shared_lift = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(
        self, satellite: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        satellite_embedding = self.satellite_projector(satellite)
        predicted_shared = self.shared_predictor(satellite)
        shared_embedding = self.shared_lift(predicted_shared)
        gate = self.gate(torch.cat([satellite_embedding, shared_embedding], dim=-1))
        prediction = self.head(
            self.fusion_norm(satellite_embedding + gate * shared_embedding)
        )
        return prediction, predicted_shared, gate


class TransformerSatelliteBottleneckStudent(nn.Module):
    """Satellite-only bottleneck student with channel-token self-attention.

    The feature archive contains one global DINOv2 vector rather than spatial
    patch tokens.  We therefore split that vector into fixed channel groups,
    embed the groups as a short sequence, and use a small Transformer encoder
    to model interactions before predicting the privileged street bottleneck.
    This is deliberately a cheap screening model; it must not be described as
    patch-level cross-view attention.
    """

    def __init__(
        self,
        satellite_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        token_count: int = 8,
        token_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
    ) -> None:
        super().__init__()
        if token_count < 2:
            raise ValueError("token_count must be at least 2.")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.satellite_dim = satellite_dim
        self.token_count = token_count
        self.chunk_dim = (satellite_dim + token_count - 1) // token_count
        self.padded_dim = self.chunk_dim * token_count

        self.input_norm = nn.LayerNorm(satellite_dim)
        self.token_projection = nn.Linear(self.chunk_dim, token_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, token_count, token_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim),
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1),
        )
        self.satellite_projector = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.bottleneck_predictor = nn.Linear(token_dim, bottleneck_dim)
        self.street_lift = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(
        self,
        satellite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(satellite)
        if self.padded_dim > self.satellite_dim:
            normalized = F.pad(normalized, (0, self.padded_dim - self.satellite_dim))
        tokens = normalized.reshape(-1, self.token_count, self.chunk_dim)
        tokens = self.token_projection(tokens) + self.position_embedding
        tokens = self.transformer(tokens)
        pool_weight = torch.softmax(self.attention_pool(tokens), dim=1)
        pooled = torch.sum(pool_weight * tokens, dim=1)

        satellite_embedding = self.satellite_projector(pooled)
        predicted_bottleneck = self.bottleneck_predictor(pooled)
        street_embedding = self.street_lift(predicted_bottleneck)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused_embedding = self.fusion_norm(
            satellite_embedding + gate * street_embedding
        )
        prediction = self.head(fused_embedding)
        return prediction, predicted_bottleneck, gate


class ProbabilisticSatelliteBottleneckStudent(nn.Module):
    """Satellite-only student predicting a Gaussian street bottleneck.

    The downstream regressor uses the conditional mean.  Log-variance is used
    only by the probabilistic auxiliary loss, so the weight-zero model remains
    a clean architecture-matched no-privilege control.
    """

    def __init__(
        self,
        satellite_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        logvar_min: float = -4.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Keep the shared modules in the same construction order as
        # SatelliteBottleneckStudent.  With the same seed, MSE training starts
        # from exactly the same shared initialization; the variance head is
        # added last and is unused by the deterministic/no-privilege controls.
        self.satellite_projector = ProjectionBlock(satellite_dim, hidden_dim, dropout)
        self.bottleneck_predictor = nn.Sequential(
            nn.LayerNorm(satellite_dim),
            nn.Linear(satellite_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.street_lift = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout)
        self.logvar_predictor = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )

    def forward(
        self,
        satellite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        satellite_embedding = self.satellite_projector(satellite)
        bottleneck_mean = self.bottleneck_predictor(satellite)
        bottleneck_logvar = self.logvar_predictor(bottleneck_mean).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        street_embedding = self.street_lift(bottleneck_mean)
        gate = self.gate(torch.cat([satellite_embedding, street_embedding], dim=-1))
        fused_embedding = self.fusion_norm(
            satellite_embedding + gate * street_embedding
        )
        prediction = self.head(fused_embedding)
        return prediction, bottleneck_mean, bottleneck_logvar, gate


def relational_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Match off-diagonal cosine-similarity relations within a mini-batch."""
    if student_embedding.shape[0] < 2:
        return student_embedding.new_zeros(())

    student_norm = F.normalize(student_embedding, dim=-1)
    teacher_norm = F.normalize(teacher_embedding.detach(), dim=-1)
    student_relation = student_norm @ student_norm.T
    teacher_relation = teacher_norm @ teacher_norm.T

    mask = ~torch.eye(
        student_relation.shape[0],
        dtype=torch.bool,
        device=student_relation.device,
    )
    return F.mse_loss(student_relation[mask], teacher_relation[mask])
