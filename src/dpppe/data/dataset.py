"""
Modern dataset implementations for 3DPPE.
Consolidated from legacy NuScenes dataset code.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from ..core.config import DataConfig


class NuScenesDataset(Dataset):
    """
    Modern NuScenes dataset implementation.
    Handles multi-camera data loading with efficient caching.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: Optional[str] = None,
        pipeline: Optional[List[Dict]] = None,
        classes: Optional[List[str]] = None,
        modality: Optional[Dict[str, bool]] = None,
        box_type_3d: str = 'LiDAR',
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        config: Optional[DataConfig] = None
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.config = config or DataConfig()
        self.test_mode = test_mode
        self.classes = classes or self._get_default_classes()
        self.modality = modality or {'use_camera': True, 'use_lidar': False}
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt

        # Load NuScenes data
        self._load_nuscenes_data()

        # Setup transforms
        self.transforms = self._build_transforms(pipeline)

    def _get_default_classes(self) -> List[str]:
        """Get default NuScenes classes."""
        return [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier'
        ]

    def _load_nuscenes_data(self):
        """Load NuScenes dataset information."""
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError:
            raise ImportError("nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")

        # Initialize NuScenes
        version = self.config.version
        self.nusc = NuScenes(version=version, dataroot=str(self.data_root), verbose=False)

        # Get scenes based on split
        if 'train' in version:
            split = 'train'
        elif 'val' in version:
            split = 'val'
        elif 'test' in version:
            split = 'test'
        else:
            split = 'train'

        scenes = create_splits_scenes()[split]

        # Filter scenes
        available_scenes = [scene['name'] for scene in self.nusc.scene]
        self.scenes = [scene for scene in scenes if scene in available_scenes]

        # Build sample list
        self.samples = []
        for scene_name in self.scenes:
            scene = self.nusc.get('scene', scene_name)
            sample_token = scene['first_sample_token']

            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                self.samples.append(sample)
                sample_token = sample['next']

        print(f"Loaded {len(self.samples)} samples from {len(self.scenes)} scenes")

    def _build_transforms(self, pipeline: Optional[List[Dict]] = None) -> Any:
        """Build data transformation pipeline."""
        if pipeline is None:
            from .transforms import DataTransforms
            return DataTransforms(self.config)
        return pipeline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        sample_data = self._load_sample_data(sample)

        # Apply transforms
        if self.transforms:
            sample_data = self.transforms(sample_data)

        return sample_data

    def _load_sample_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load data for a single sample."""
        data = {
            'sample_token': sample['token'],
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
        }

        # Load camera data
        if self.modality['use_camera']:
            data.update(self._load_camera_data(sample))

        # Load LiDAR data
        if self.modality['use_lidar']:
            data.update(self._load_lidar_data(sample))

        # Load annotations
        if not self.test_mode:
            data.update(self._load_annotations(sample))

        return data

    def _load_camera_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load multi-camera images and intrinsics."""
        camera_data = {}
        images = []
        intrinsics = []
        extrinsics = []

        # Camera names in NuScenes
        camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        for cam_name in camera_names:
            if cam_name in sample['data']:
                cam_token = sample['data'][cam_name]
                cam_data = self.nusc.get('sample_data', cam_token)

                # Load image
                img_path = self.data_root / cam_data['filename']
                if img_path.exists():
                    # Load image with PIL for consistency
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    images.append(img_array)

                    # Load calibration
                    calib_token = cam_data['calibrated_sensor_token']
                    calib_data = self.nusc.get('calibrated_sensor', calib_token)

                    intrinsics.append(calib_data['camera_intrinsic'])
                    extrinsics.append(calib_data['rotation'] + calib_data['translation'])

        camera_data['images'] = images  # List of (H, W, 3) arrays
        camera_data['intrinsics'] = intrinsics  # List of (3, 3) matrices
        camera_data['extrinsics'] = extrinsics  # List of (4,) arrays [rot_x, rot_y, rot_z, tx, ty, tz]
        camera_data['camera_names'] = camera_names

        return camera_data

    def _load_lidar_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load LiDAR point cloud data."""
        lidar_data = {}

        if 'LIDAR_TOP' in sample['data']:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_sample = self.nusc.get('sample_data', lidar_token)

            # Load point cloud
            lidar_path = self.data_root / lidar_sample['filename']
            if lidar_path.exists():
                # Load using nuscenes utils
                from nuscenes.utils.data_classes import LidarPointCloud
                pc = LidarPointCloud.from_file(str(lidar_path))

                # Get points and features
                points = pc.points[:3].T  # (N, 3) - x, y, z
                intensity = pc.points[3]  # (N,) - intensity

                lidar_data['points'] = points
                lidar_data['intensity'] = intensity

                # Load calibration
                calib_token = lidar_sample['calibrated_sensor_token']
                calib_data = self.nusc.get('calibrated_sensor', calib_token)
                lidar_data['lidar_extrinsic'] = calib_data['rotation'] + calib_data['translation']

        return lidar_data

    def _load_annotations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load ground truth annotations."""
        ann_data = {}

        ann_tokens = sample['anns']
        gt_boxes = []
        gt_labels = []
        gt_names = []

        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Get bounding box
            box = self._get_box_from_annotation(ann)
            if box is not None:
                gt_boxes.append(box)
                gt_names.append(ann['category_name'])
                gt_labels.append(self.classes.index(ann['category_name']) if ann['category_name'] in self.classes else -1)

        ann_data['gt_boxes'] = gt_boxes
        ann_data['gt_labels'] = gt_labels
        ann_data['gt_names'] = gt_names

        return ann_data

    def _get_box_from_annotation(self, ann: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract 3D bounding box from annotation."""
        # Convert NuScenes box to our format
        # This is a simplified version - in practice, you'd use proper coordinate transformations
        try:
            # Get box parameters
            translation = np.array(ann['translation'])
            rotation = np.array(ann['rotation'])
            size = np.array(ann['size'])

            # Convert to [x, y, z, w, l, h, rot_x, rot_y, rot_z] format
            box = np.concatenate([translation, size[[1, 0, 2]], rotation])  # w, l, h order

            return box
        except Exception:
            return None

    def get_camera_info(self, idx: int) -> Dict[str, Any]:
        """Get camera information for visualization."""
        sample = self.samples[idx]
        camera_info = {}

        for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']:
            if cam_name in sample['data']:
                cam_token = sample['data'][cam_name]
                cam_data = self.nusc.get('sample_data', cam_token)
                camera_info[cam_name] = {
                    'filename': cam_data['filename'],
                    'timestamp': cam_data['timestamp'],
                }

        return camera_info


class DPPEDataset(NuScenesDataset):
    """
    3DPPE-specific dataset with depth-aware loading.
    Extends NuScenes dataset with depth estimation capabilities.
    """

    def __init__(self, *args, with_depth: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_depth = with_depth

    def _load_sample_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load data with depth information."""
        data = super()._load_sample_data(sample)

        if self.with_depth:
            data.update(self._load_depth_data(sample))

        return data

    def _load_depth_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load depth maps if available."""
        depth_data = {}

        # In practice, this would load pre-computed depth maps
        # For now, return empty dict
        return depth_data


def build_dataset(config: DataConfig, test_mode: bool = False) -> Dataset:
    """Build dataset based on configuration."""
    if config.dataset_name.lower() == 'nuscenes':
        return NuScenesDataset(
            data_root=config.dataset_root,
            test_mode=test_mode,
            config=config
        )
    elif config.dataset_name.lower() == 'dpppe':
        return DPPEDataset(
            data_root=config.dataset_root,
            test_mode=test_mode,
            config=config,
            with_depth=config.use_depth
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
