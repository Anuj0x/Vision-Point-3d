"""
Depth estimation modules for 3DPPE.
Consolidated from legacy depth estimation code.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DepthEncoder(nn.Module):
    """
    Depth encoder for estimating depth from multi-view images.
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, 1),  # Output depth map
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C, H, W) - Multi-view features

        Returns:
            depth: (B, N, 1, H, W) - Estimated depth maps
        """
        B, N, C, H, W = x.shape

        # Process each view separately
        x_flat = x.view(B * N, C, H, W)
        depth_flat = self.encoder(x_flat)
        depth = depth_flat.view(B, N, 1, H, W)

        return depth


class CategoricalDepthHead(nn.Module):
    """
    Categorical depth estimation head.
    Predicts depth categories instead of continuous values.
    """

    def __init__(self, in_channels: int = 256, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, num_bins, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C, H, W)

        Returns:
            depth_logits: (B, N, num_bins, H, W)
        """
        B, N, C, H, W = x.shape

        x_flat = x.view(B * N, C, H, W)
        logits_flat = self.head(x_flat)
        logits = logits_flat.view(B, N, self.num_bins, H, W)

        return logits

    def predict_depth(self, logits: Tensor, depth_bins: Tensor) -> Tensor:
        """
        Convert logits to depth predictions.

        Args:
            logits: (B, N, num_bins, H, W)
            depth_bins: (num_bins,) - Depth bin centers

        Returns:
            depth: (B, N, 1, H, W)
        """
        probs = F.softmax(logits, dim=2)  # (B, N, num_bins, H, W)

        # Weighted sum of depth bins
        depth_bins = depth_bins.view(1, 1, -1, 1, 1)  # (1, 1, num_bins, 1, 1)
        depth = torch.sum(probs * depth_bins, dim=2, keepdim=True)  # (B, N, 1, H, W)

        return depth


class HybridDepthModule(nn.Module):
    """
    Hybrid depth module combining direct and categorical depth estimation.
    This is a key component of 3DPPE.
    """

    def __init__(self, in_channels: int = 256, num_bins: int = 64):
        super().__init__()

        self.direct_depth_head = DepthEncoder(in_channels, in_channels // 2)
        self.categorical_depth_head = CategoricalDepthHead(in_channels, num_bins)

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + 1 + num_bins, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Depth bins (learnable)
        self.depth_bins = nn.Parameter(torch.linspace(0.1, 100.0, num_bins))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C, H, W) - Multi-view features

        Returns:
            fused_features: (B, N, C, H, W) - Features fused with depth information
        """
        # Direct depth estimation
        direct_depth = self.direct_depth_head(x)  # (B, N, 1, H, W)

        # Categorical depth estimation
        cat_logits = self.categorical_depth_head(x)  # (B, N, num_bins, H, W)
        cat_depth = self.categorical_depth_head.predict_depth(cat_logits, self.depth_bins)  # (B, N, 1, H, W)

        # Combine direct and categorical depth
        combined_depth = torch.cat([direct_depth, cat_depth], dim=2)  # (B, N, 2, H, W)

        # Concatenate with original features
        B, N, C, H, W = x.shape
        x_reshaped = x.view(B * N, C, H, W)
        depth_logits_reshaped = cat_logits.view(B * N, self.categorical_depth_head.num_bins, H, W)
        combined_depth_reshaped = combined_depth.view(B * N, 2, H, W)

        # Concatenate along channel dimension
        fused = torch.cat([x_reshaped, combined_depth_reshaped, depth_logits_reshaped], dim=1)

        # Fuse
        fused_features = self.fusion(fused)

        # Reshape back
        fused_features = fused_features.view(B, N, C, H, W)

        return fused_features


class ContextEncoder(nn.Module):
    """
    Context encoder for depth-aware feature enhancement.
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        super().__init__()

        self.context_encoder = nn.Sequential(
            # Multi-scale context
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C, H, W)

        Returns:
            context_features: (B, N, C, H, W)
        """
        B, N, C, H, W = x.shape

        x_flat = x.view(B * N, C, H, W)
        context_flat = self.context_encoder(x_flat)
        context = context_flat.view(B, N, C, H, W)

        return context
