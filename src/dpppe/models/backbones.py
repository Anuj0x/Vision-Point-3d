"""
Modern backbone implementations.
Consolidated from legacy VoVNet and other backbone code.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch import Tensor


class VoVNetBlock(nn.Module):
    """VoVNet block with multiple convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_conv: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act_type: str = 'relu'
    ):
        super().__init__()

        layers = []
        for i in range(num_conv):
            in_c = in_channels if i == 0 else mid_channels

            conv = nn.Conv2d(
                in_c, mid_channels, kernel_size, stride=stride if i == 0 else 1,
                padding=dilation, dilation=dilation, groups=groups, bias=False
            )
            layers.append(conv)

            if act_type == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif act_type == 'swish':
                layers.append(nn.SiLU(inplace=True))

        # Concatenate all intermediate features
        concat_conv = nn.Conv2d(
            mid_channels * num_conv, out_channels, 1, bias=False
        )
        layers.append(concat_conv)

        self.layers = nn.Sequential(*layers)

        # Downsample path if stride > 1
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.bn = nn.BatchNorm2d(out_channels)

        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'swish':
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.layers(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.bn(out)
        out = self.act(out)

        return out


class VoVNetBackbone(nn.Module):
    """
    Modern VoVNet backbone implementation.
    Based on VoVNet-99 configuration for optimal performance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        # Default VoVNet-99 configuration
        self.config = {
            'stem_channels': 64,
            'stage_channels': [128, 160, 192, 224],
            'stage_blocks': [1, 3, 9, 3],
            'stage_mid_channels': [64, 80, 96, 112],
            'stage_conv_nums': [3, 3, 3, 3],
            'out_indices': (0, 1, 2, 3),
            **config
        }

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.config['stem_channels']//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.config['stem_channels']//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.config['stem_channels']//2, self.config['stem_channels'], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.config['stem_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.config['stem_channels'], self.config['stem_channels'], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.config['stem_channels']),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        in_channels = self.config['stem_channels']

        for i, (out_channels, num_blocks, mid_channels, num_conv) in enumerate(zip(
            self.config['stage_channels'],
            self.config['stage_blocks'],
            self.config['stage_mid_channels'],
            self.config['stage_conv_nums']
        )):
            stage = nn.Sequential()

            # First block with stride 2 (except first stage)
            stride = 2 if i > 0 else 1
            block = VoVNetBlock(
                in_channels, mid_channels, out_channels, num_conv,
                stride=stride
            )
            stage.add_module(f'block_0', block)

            # Remaining blocks
            for j in range(1, num_blocks):
                block = VoVNetBlock(
                    out_channels, mid_channels, out_channels, num_conv,
                    stride=1
                )
                stage.add_module(f'block_{j}', block)

            self.stages.append(stage)
            in_channels = out_channels

        self.out_indices = self.config['out_indices']
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using modern initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature maps at different scales
        """
        outputs = []

        x = self.stem(x)  # (B, 64, H/4, W/4)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)

        return outputs  # Usually 4 feature maps at different scales

    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained ImageNet weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle key mismatches (MMDetection vs our implementation)
        model_dict = self.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            # Remove module prefix if present
            if k.startswith('module.'):
                k = k[7:]

            # Map MMDetection keys to our keys
            if k.startswith('backbone.'):
                k = k[9:]  # Remove backbone prefix

            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} backbone parameters")


class ResNetBackbone(nn.Module):
    """Alternative ResNet backbone for comparison."""

    def __init__(self, depth: int = 50, out_indices: tuple = (1, 2, 3, 4)):
        super().__init__()

        if depth == 50:
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
        elif depth == 101:
            from torchvision.models import resnet101
            self.backbone = resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.out_indices = out_indices

        # Remove classification head
        self.backbone.fc = nn.Identity()

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass returning intermediate features."""
        outputs = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1
        x = self.backbone.layer1(x)
        if 0 in self.out_indices:
            outputs.append(x)

        # Layer 2
        x = self.backbone.layer2(x)
        if 1 in self.out_indices:
            outputs.append(x)

        # Layer 3
        x = self.backbone.layer3(x)
        if 2 in self.out_indices:
            outputs.append(x)

        # Layer 4
        x = self.backbone.layer4(x)
        if 3 in self.out_indices:
            outputs.append(x)

        return outputs
