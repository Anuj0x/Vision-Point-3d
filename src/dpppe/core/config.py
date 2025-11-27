"""
Modern configuration system using Hydra and OmegaConf.
Provides type-safe, hierarchical configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Configuration for the 3DPPE model architecture."""

    # Backbone configuration
    backbone_type: str = "VoVNet"
    backbone_config: Dict[str, Any] = field(default_factory=dict)

    # Transformer configuration
    num_queries: int = 900
    num_decoder_layers: int = 6
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    activation: str = "relu"

    # Depth estimation
    use_depth: bool = True
    depth_channels: int = 64
    depth_levels: List[int] = field(default_factory=lambda: [16, 32])

    # 3D positional encoding
    pe_temperature: float = 10000.0
    pe_scale: float = 1.0

    # Loss weights
    cls_loss_weight: float = 2.0
    bbox_loss_weight: float = 5.0
    depth_loss_weight: float = 1.0


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Dataset
    dataset_name: str = "nuscenes"
    dataset_root: str = "./data/nuscenes"
    version: str = "v1.0-trainval"

    # Data loading
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Image preprocessing
    img_size: List[int] = field(default_factory=lambda: [900, 1600])
    img_norm_cfg: Dict[str, List[float]] = field(default_factory=lambda: {
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True
    })

    # Augmentation
    use_aug: bool = True
    random_flip: float = 0.5
    random_crop: Optional[List[float]] = None
    color_jitter: Optional[Dict[str, float]] = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Optimizer
    optimizer: str = "AdamW"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # Scheduler
    scheduler: str = "StepLR"
    step_size: int = 30
    gamma: float = 0.1

    # Training loop
    max_epochs: int = 100
    gradient_clip_val: float = 0.1
    accumulate_grad_batches: int = 1

    # Mixed precision
    precision: str = "16-mixed"

    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val/mAP"
    mode: str = "max"


@dataclass
class DPPEConfig:
    """Main configuration class for 3DPPE."""

    # Core components
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment tracking
    experiment_name: str = "3DPPE_experiment"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # Device and distributed training
    device: str = "auto"  # auto, cpu, cuda, mps
    accelerator: str = "auto"
    devices: Union[int, str, List[int]] = "auto"
    strategy: str = "auto"

    # Random seed
    seed: Optional[int] = 42

    # Additional settings
    debug: bool = False
    verbose: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create necessary directories
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.checkpoint_dir).mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "DPPEConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_object(cfg)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, str(output_path))

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        cfg = OmegaConf.structured(self)
        cfg = OmegaConf.merge(cfg, updates)
        updated_config = OmegaConf.to_object(cfg)
        self.__dict__.update(updated_config.__dict__)


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=DPPEConfig)


def get_config(config_path: Optional[str] = None) -> DPPEConfig:
    """Get configuration, optionally from file."""
    if config_path is not None:
        return DPPEConfig.from_yaml(config_path)
    return DPPEConfig()
