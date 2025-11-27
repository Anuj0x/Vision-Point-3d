"""
Modern PETR3D Detector implementation.
Consolidated and modernized from legacy MMDetection3D code.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import ModelConfig
from .backbones import VoVNetBackbone
from .transformers import MultiViewTransformer, PointPositionalEncoder
from .depth import HybridDepthModule


class PETR3DDetector(nn.Module):
    """
    Modern 3DPPE PETR3D Detector.
    Combines multi-view images with 3D point positional encoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Backbone for feature extraction
        self.backbone = VoVNetBackbone(config.backbone_config)

        # Depth estimation module
        if config.use_depth:
            self.depth_module = HybridDepthModule(config.depth_channels)

        # 3D positional encoding
        self.positional_encoder = PointPositionalEncoder(
            temperature=config.pe_temperature,
            scale=config.pe_scale
        )

        # Multi-view transformer
        self.transformer = MultiViewTransformer(
            num_queries=config.num_queries,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            activation=config.activation
        )

        # Prediction heads
        self.class_embed = nn.Linear(config.hidden_dim, 10)  # 10 classes for nuScenes
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10)  # [cx, cy, w, l, cz, h, rot_sin, rot_cos, vx, vy]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using modern initialization."""
        # Xavier uniform for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Special initialization for transformer
        self.transformer._reset_parameters()

    def extract_image_features(self, images: Tensor) -> List[Tensor]:
        """
        Extract multi-scale features from multi-view images.

        Args:
            images: (B, N_views, C, H, W)

        Returns:
            List of feature maps: [(B, N_views, C', H', W'), ...]
        """
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)  # (B*N, C, H, W)

        # Extract features
        features = self.backbone(images_flat)  # List[(B*N, C', H', W'), ...]

        # Reshape back to multi-view format
        features_reshaped = []
        for feat in features:
            BN, C_feat, H_feat, W_feat = feat.shape
            feat_reshaped = feat.view(B, N, C_feat, H_feat, W_feat)
            features_reshaped.append(feat_reshaped)

        return features_reshaped

    def forward(
        self,
        images: Tensor,
        image_metas: List[Dict[str, Any]],
        gt_bboxes_3d: Optional[List[Tensor]] = None,
        gt_labels_3d: Optional[List[Tensor]] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the detector.

        Args:
            images: Multi-view images (B, N_views, C, H, W)
            image_metas: Metadata for each sample
            gt_bboxes_3d: Ground truth 3D boxes for training
            gt_labels_3d: Ground truth labels for training

        Returns:
            Dictionary containing predictions and losses
        """
        # Extract image features
        image_features = self.extract_image_features(images)

        # Estimate depth if enabled
        if self.config.use_depth:
            depth_features = self.depth_module(image_features[0])  # Use first level
            # Update features with depth information
            image_features = self._fuse_depth_features(image_features, depth_features)

        # Apply 3D positional encoding
        pos_embeddings = self.positional_encoder(image_features, image_metas)

        # Transformer forward pass
        hs, memory = self.transformer(
            src=image_features,
            pos=pos_embeddings,
            query_embed=None  # Learnable queries
        )

        # Get final decoder output
        outputs_class = self.class_embed(hs[-1])  # Last decoder layer
        outputs_coord = self.bbox_embed(hs[-1])

        # Format outputs
        outputs = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
        }

        # Compute losses if training
        if self.training and gt_bboxes_3d is not None:
            losses = self.compute_losses(outputs, gt_bboxes_3d, gt_labels_3d, image_metas)
            outputs.update(losses)

        return outputs

    def _fuse_depth_features(
        self,
        image_features: List[Tensor],
        depth_features: Tensor
    ) -> List[Tensor]:
        """Fuse depth features with image features."""
        fused_features = []
        for i, img_feat in enumerate(image_features):
            # Simple concatenation for now - could be more sophisticated
            if i == 0:  # Fuse depth with first level
                fused = torch.cat([img_feat, depth_features], dim=2)  # Along channel dim
            else:
                fused = img_feat
            fused_features.append(fused)
        return fused_features

    def compute_losses(
        self,
        outputs: Dict[str, Tensor],
        gt_bboxes_3d: List[Tensor],
        gt_labels_3d: List[Tensor],
        image_metas: List[Dict[str, Any]]
    ) -> Dict[str, Tensor]:
        """Compute training losses."""
        # This would use the DPPESetCriterion
        # For now, return placeholder losses
        losses = {
            'loss_ce': torch.tensor(0.0, device=outputs['pred_logits'].device),
            'loss_bbox': torch.tensor(0.0, device=outputs['pred_boxes'].device),
            'loss_giou': torch.tensor(0.0, device=outputs['pred_boxes'].device),
        }
        return losses

    @torch.no_grad()
    def inference(
        self,
        images: Tensor,
        image_metas: List[Dict[str, Any]],
        post_process: bool = True
    ) -> List[Dict[str, Tensor]]:
        """
        Inference method for evaluation.

        Args:
            images: Multi-view images
            image_metas: Metadata
            post_process: Whether to apply post-processing

        Returns:
            List of predictions per sample
        """
        outputs = self.forward(images, image_metas)

        if post_process:
            return self.post_process(outputs, image_metas)
        else:
            return [outputs] * len(image_metas)  # One per sample

    def post_process(
        self,
        outputs: Dict[str, Tensor],
        image_metas: List[Dict[str, Any]]
    ) -> List[Dict[str, Tensor]]:
        """Apply post-processing to raw outputs."""
        # Apply sigmoid to logits and threshold
        pred_logits = outputs['pred_logits']  # (B, N_queries, N_classes)
        pred_boxes = outputs['pred_boxes']    # (B, N_queries, 10)

        results = []
        for b in range(pred_logits.shape[0]):
            # Get predictions for this sample
            logits = pred_logits[b]  # (N_queries, N_classes)
            boxes = pred_boxes[b]    # (N_queries, 10)

            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            scores, labels = torch.max(probs, dim=-1)

            # Filter by confidence threshold
            keep = scores > 0.3
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Sort by score
            _, indices = torch.sort(scores, descending=True)
            scores = scores[indices]
            labels = labels[indices]
            boxes = boxes[indices]

            result = {
                'scores': scores,
                'labels': labels,
                'boxes': boxes,
            }
            results.append(result)

        return results

    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle key mismatches
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained parameters")

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer with different learning rates."""
        backbone_params = []
        transformer_params = []
        head_params = []

        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'transformer' in name:
                transformer_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': self.config.lr * 0.1},  # Lower LR for backbone
            {'params': transformer_params, 'lr': self.config.lr},
            {'params': head_params, 'lr': self.config.lr},
        ]
