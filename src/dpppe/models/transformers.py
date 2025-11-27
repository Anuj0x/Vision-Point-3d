"""
Modern transformer implementations for 3DPPE.
Consolidated from legacy transformer code with modern PyTorch features.
"""

from typing import List, Dict, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position embedding used in the original transformer paper.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, N, C, H, W)
            mask: (B, H, W)
        """
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PointPositionalEncoder(nn.Module):
    """
    3D Point Positional Encoding for multi-camera 3D detection.
    Encodes 3D point positions into positional embeddings.
    """

    def __init__(self, temperature: float = 10000.0, scale: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.scale = scale

    def forward(self, features: List[Tensor], image_metas: List[Dict[str, Any]]) -> List[Tensor]:
        """
        Generate 3D point positional encodings.

        Args:
            features: List of feature maps [(B, N_views, C, H, W), ...]
            image_metas: Metadata containing camera parameters

        Returns:
            List of positional embeddings
        """
        pos_embeddings = []

        for level, feat in enumerate(features):
            B, N, C, H, W = feat.shape

            # Get camera intrinsics and extrinsics from metadata
            # This is a simplified version - in practice, you'd need proper 3D projection
            intrinsics = self._get_intrinsics(image_metas, B, N)
            extrinsics = self._get_extrinsics(image_metas, B, N)

            # Generate 3D point positions for each pixel
            points_3d = self._generate_3d_points(H, W, intrinsics, extrinsics, feat.device)

            # Convert 3D points to positional embeddings
            pos_embed = self._points_to_embeddings(points_3d, C)

            pos_embeddings.append(pos_embed)

        return pos_embeddings

    def _get_intrinsics(self, image_metas: List[Dict], batch_size: int, num_views: int) -> Tensor:
        """Extract camera intrinsics from metadata."""
        # Placeholder - in practice, extract from image_metas
        return torch.eye(3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)

    def _get_extrinsics(self, image_metas: List[Dict], batch_size: int, num_views: int) -> Tensor:
        """Extract camera extrinsics from metadata."""
        # Placeholder - in practice, extract from image_metas
        return torch.eye(4, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)

    def _generate_3d_points(self, H: int, W: int, intrinsics: Tensor, extrinsics: Tensor, device: torch.device) -> Tensor:
        """Generate 3D point positions for image pixels."""
        # Create pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # Convert to camera coordinates (simplified)
        # In practice, this would involve proper depth estimation and 3D projection
        points_3d = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1)
        points_3d = points_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)

        return points_3d

    def _points_to_embeddings(self, points_3d: Tensor, embed_dim: int) -> Tensor:
        """Convert 3D points to positional embeddings."""
        B, N, H, W, _ = points_3d.shape

        # Flatten spatial dimensions
        points_flat = points_3d.view(B * N * H * W, 3)  # (B*N*H*W, 3)

        # Apply sinusoidal encoding
        pos_embed = self._sinusoidal_encoding(points_flat, embed_dim)

        # Reshape back
        pos_embed = pos_embed.view(B, N, H, W, embed_dim).permute(0, 4, 1, 2, 3)  # (B, C, N, H, W)

        return pos_embed

    def _sinusoidal_encoding(self, points: Tensor, embed_dim: int) -> Tensor:
        """Apply sinusoidal positional encoding to 3D points."""
        # points: (N_points, 3)
        # embed_dim must be even

        embeddings = []
        for i in range(embed_dim // 2):
            for coord in range(3):  # x, y, z coordinates
                angle = points[:, coord] / (self.temperature ** (2 * i / embed_dim))
                embeddings.append(torch.sin(angle))
                embeddings.append(torch.cos(angle))

        return torch.stack(embeddings, dim=-1) * self.scale


class MultiHeadAttention(nn.Module):
    """Modern multi-head attention with efficient implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: (B, N, C)
            key: (B, N, C)
            value: (B, N, C)
        """
        B, N, C = query.shape

        # Linear projections
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        output = self.out_proj(attn_output)

        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + attn_out)

        # Feed-forward
        ff_out = self.ff(src)
        src = self.norm2(src + ff_out)

        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self_attn_out)

        # Cross-attention
        cross_attn_out, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + cross_attn_out)

        # Feed-forward
        ff_out = self.ff(tgt)
        tgt = self.norm3(tgt + ff_out)

        return tgt


class MultiViewTransformer(nn.Module):
    """
    Multi-view transformer for 3DPPE.
    Handles multiple camera views with 3D positional encoding.
    """

    def __init__(
        self,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Input projection layers for different scales
        self.input_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(4)  # 4 feature scales
        ])

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: List[Tensor],
        pos: List[Tensor],
        query_embed: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], Tensor]:
        """
        Args:
            src: List of multi-view features [(B, N_views, C, H, W), ...]
            pos: List of positional embeddings
            query_embed: Optional query embeddings

        Returns:
            hs: List of decoder outputs per layer
            memory: Encoder memory
        """
        # Project inputs to hidden dimension
        src_proj = []
        pos_proj = []

        for i, (s, p) in enumerate(zip(src, pos)):
            B, N, C, H, W = s.shape

            # Flatten spatial and view dimensions
            s_flat = s.view(B, N * H * W, C)  # (B, N*H*W, C)
            p_flat = p.view(B, N * H * W, C)  # (B, N*H*W, C)

            # Project to hidden dimension
            s_proj = self.input_proj[i](s_flat.transpose(1, 2)).transpose(1, 2)
            src_proj.append(s_proj + p_flat)

        # Concatenate all levels
        src_cat = torch.cat(src_proj, dim=1)  # (B, total_tokens, C)

        # Encoder
        memory = src_cat
        for encoder_layer in self.encoder:
            memory = encoder_layer(memory)

        # Decoder
        if query_embed is None:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        tgt = torch.zeros_like(query_embed)
        hs = []

        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory)
            hs.append(self.output_proj(tgt))

        return hs, memory
