"""
Deformable Attention Module for License Plate Character Recognition

Fixes 0% sequence accuracy by preserving spatial structure and learning
character-specific spatial locations instead of flattening feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class DeformableCharacterAttention(nn.Module):
    """Deformable attention that learns where each character is located spatially."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_chars: int = 10,
        feature_dim: int = 144,
        num_scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_chars = num_chars
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        self.char_queries = nn.Parameter(torch.randn(num_chars, hidden_dim))

        # Key innovation: learn WHERE each character is located
        self.offset_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2),
                nn.Tanh()  # [-1, 1] for grid_sample
            ) for _ in range(num_chars)
        ])

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.char_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 35)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.char_queries)

        # Critical: initialize offsets to reasonable character positions
        for i, offset_net in enumerate(self.offset_networks):
            with torch.no_grad():
                init_x = -0.8 + (1.6 * i / (self.num_chars - 1)) if self.num_chars > 1 else 0.0
                init_y = 0.0
                offset_net[-2].bias[0] = init_x
                offset_net[-2].bias[1] = init_y
                offset_net[-2].weight.data *= 0.1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def spatial_sample(self, feature_map: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample features at learned spatial locations using grid_sample."""
        batch_size = feature_map.shape[0]
        grid = offset.view(batch_size, 1, 1, 2)

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return sampled.squeeze(-1).squeeze(-1)

    def multi_scale_sampling(self, feature_maps: List[torch.Tensor], offset: torch.Tensor) -> torch.Tensor:
        """Sample from P3/P4/P5 at same spatial location and fuse."""
        scale_features = []

        for feature_map in feature_maps:
            sampled = self.spatial_sample(feature_map, offset)
            projected = self.feature_proj(sampled)
            scale_features.append(projected)

        multi_scale = torch.cat(scale_features, dim=-1)
        return self.scale_fusion(multi_scale)

    def forward(self, feature_maps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = feature_maps[0].shape[0]
        all_char_features = []
        all_offsets = []

        for char_idx in range(self.num_chars):
            char_query = self.char_queries[char_idx].unsqueeze(0).expand(batch_size, -1)

            # Learn WHERE this character is located
            offset = self.offset_networks[char_idx](char_query)
            all_offsets.append(offset)

            # Sample features at learned location
            char_features = self.multi_scale_sampling(feature_maps, offset)
            char_features = char_features + char_query
            char_features = self.layer_norm1(char_features)
            all_char_features.append(char_features)

        char_features = torch.stack(all_char_features, dim=1)
        attention_offsets = torch.stack(all_offsets, dim=1)

        attended_features, _ = self.self_attention(char_features, char_features, char_features)
        char_features = char_features + attended_features
        char_features = self.layer_norm2(char_features)

        char_logits = self.char_classifier(char_features)
        return char_logits, attention_offsets

    def visualize_attention(self, image: torch.Tensor, attention_offsets: torch.Tensor, save_path: Optional[str] = None):
        """Visualize where each character position is attending."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        batch_size, _, img_h, img_w = image.shape

        for b in range(min(batch_size, 4)):
            img = image[b].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())

            plt.figure(figsize=(12, 4))
            plt.imshow(img)

            offsets = attention_offsets[b].cpu().numpy()

            for char_idx, (x, y) in enumerate(offsets):
                img_x = (x + 1) * img_w / 2
                img_y = (y + 1) * img_h / 2
                plt.plot(img_x, img_y, 'ro', markersize=8, alpha=0.8)
                plt.text(img_x, img_y - 10, f'{char_idx+1}', color='red', fontsize=12, ha='center', weight='bold')

            plt.title(f'Character Attention Locations (Image {b+1})')
            plt.axis('off')

            if save_path:
                plt.savefig(f"{save_path}_batch_{b}.png", dpi=150, bbox_inches='tight')
            plt.close()


class SpatialDiversityLoss(nn.Module):
    """Prevents characters from attending to same spatial location."""

    def __init__(self, min_distance: float = 0.1, loss_weight: float = 0.1):
        super().__init__()
        self.min_distance = min_distance
        self.loss_weight = loss_weight

    def forward(self, attention_offsets: torch.Tensor) -> torch.Tensor:
        batch_size, num_chars, _ = attention_offsets.shape
        total_loss = 0.0
        num_pairs = 0

        for i in range(num_chars):
            for j in range(i + 1, num_chars):
                diff = attention_offsets[:, i, :] - attention_offsets[:, j, :]
                distance = torch.norm(diff, dim=-1)
                penalty = torch.clamp(self.min_distance - distance, min=0.0)
                total_loss += penalty.mean()
                num_pairs += 1

        diversity_loss = total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
        return self.loss_weight * diversity_loss


def test_deformable_attention():
    """Test deformable attention module."""
    batch_size = 2
    feature_maps = [
        torch.randn(batch_size, 144, 28, 28),  # P3
        torch.randn(batch_size, 144, 14, 14),  # P4
        torch.randn(batch_size, 144, 7, 7),    # P5
    ]

    deformable_attn = DeformableCharacterAttention(hidden_dim=256, num_heads=8, num_chars=10, feature_dim=144)

    with torch.no_grad():
        char_logits, attention_offsets = deformable_attn(feature_maps)

    print(f"char_logits shape: {char_logits.shape}")
    print(f"attention_offsets shape: {attention_offsets.shape}")
    print(f"offset range: [{attention_offsets.min():.3f}, {attention_offsets.max():.3f}]")

    diversity_loss_fn = SpatialDiversityLoss()
    diversity_loss = diversity_loss_fn(attention_offsets)
    print(f"diversity loss: {diversity_loss.item():.6f}")

    return deformable_attn, char_logits, attention_offsets


if __name__ == "__main__":
    test_deformable_attention()