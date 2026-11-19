"""
PROBA-V MISR Data Loader

Loads the official ESA PROBA-V Super-Resolution challenge dataset.
Supports both original challenge format and PROBA-V-ref variant.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ProbaVScene:
    """Single PROBA-V scene with LR stack and HR reference."""
    scene_id: str
    band: str  # RED or NIR
    lr_images: List[np.ndarray]  # List of [H, W] arrays (128x128)
    lr_quality_masks: List[np.ndarray]  # List of [H, W] masks
    hr_image: Optional[np.ndarray]  # [H, W] array (384x384) or None for test
    hr_status_mask: Optional[np.ndarray]  # [H, W] mask
    split: str  # train or test


class ProbaVLoader:
    """Load and process PROBA-V MISR dataset."""

    def __init__(self, config_path: str = "configs/data/probav.yaml"):
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(self.config['structure']['root'])
        self.bands = self.config['sensor']['challenge_bands']
        self.scale = self.config['images']['scale_factor']
        self.max_lr = self.config['processing']['stack']['max_lr_per_scene']

        # Normalization
        self.max_value = self.config['processing']['normalization']['max_value']

    def list_scenes(self, split: str, band: str) -> List[str]:
        """List available scene IDs for a split and band."""
        band_dir = self.data_dir / split / band
        if not band_dir.exists():
            return []

        scenes = []
        for item in band_dir.iterdir():
            if item.is_dir() and item.name.startswith('imgset'):
                scenes.append(item.name)
        return sorted(scenes)

    def load_scene(self, scene_id: str, band: str, split: str) -> ProbaVScene:
        """Load a single scene (LR stack + HR reference).

        Masks are normalized to {0, 1}: the archive stores clear=255 in
        SM/QM PNGs; loader converts any nonzero value to 1.
        """
        scene_dir = self.data_dir / split / band / scene_id

        # Load LR images
        lr_images = []
        lr_masks = []

        lr_files = sorted(scene_dir.glob('LR*.png'))
        qm_files = sorted(scene_dir.glob('QM*.png'))

        for lr_file, qm_file in zip(lr_files, qm_files):
            lr_img = cv2.imread(str(lr_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
            qm_img = cv2.imread(str(qm_file), cv2.IMREAD_UNCHANGED).astype(np.uint8)

            # Normalize to [0, 1]
            lr_img = lr_img / self.max_value
            # Mask: clear=255 in PNG -> {0, 1}
            qm_img = (qm_img > 0).astype(np.uint8)
            lr_images.append(lr_img)
            lr_masks.append(qm_img)

        # Load HR if available (train split)
        hr_image = None
        hr_mask = None
        hr_file = scene_dir / 'HR.png'
        sm_file = scene_dir / 'SM.png'

        if hr_file.exists():
            hr_img = cv2.imread(str(hr_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
            hr_image = hr_img / self.max_value
            hr_mask = cv2.imread(str(sm_file), cv2.IMREAD_UNCHANGED).astype(np.uint8)
            hr_mask = (hr_mask > 0).astype(np.uint8)
        elif sm_file.exists():
            # Test scenes: no HR (challenge), but SM is the evaluation mask
            hr_mask = cv2.imread(str(sm_file), cv2.IMREAD_UNCHANGED).astype(np.uint8)
            hr_mask = (hr_mask > 0).astype(np.uint8)

        return ProbaVScene(
            scene_id=scene_id,
            band=band,
            lr_images=lr_images,
            lr_quality_masks=lr_masks,
            hr_image=hr_image,
            hr_status_mask=hr_mask,
            split=split
        )

    def load_all_scenes(self, split: str) -> Dict[str, Dict[str, ProbaVScene]]:
        """Load all scenes for a split, organized by scene_id and band."""
        scenes = {}

        for band in self.bands:
            scene_ids = self.list_scenes(split, band)
            for scene_id in scene_ids:
                if scene_id not in scenes:
                    scenes[scene_id] = {}
                scenes[scene_id][band] = self.load_scene(scene_id, band, split)

        return scenes

    def get_scene_stack(self, scenes: Dict, scene_id: str) -> Tuple[List[np.ndarray], Optional[np.ndarray], str]:
        """
        Get the LR stack and HR reference for a single-band scene.

        The archive partitions imgset IDs by band (each ID belongs to
        exactly one band), so a scene is single-band by construction.
        Returns: (lr_stack [N, 1, 128, 128], hr [1, 384, 384] or None, split)
        """
        if scene_id not in scenes:
            raise ValueError(f"Scene {scene_id} not found")

        band_scenes = scenes[scene_id]
        if not band_scenes:
            raise ValueError(f"Scene {scene_id} has no band data")

        band = next(iter(band_scenes))
        scene = band_scenes[band]

        # Stack LR images as single-channel: [N, 1, 128, 128]
        lr_stack = [np.stack([img], axis=0) for img in scene.lr_images]

        hr_ref = None
        if scene.hr_image is not None:
            hr_ref = np.stack([scene.hr_image], axis=0)  # [1, 384, 384]

        return lr_stack, hr_ref, scene.split

    def prepare_training_pairs(self, scenes: Dict) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare training pairs for single-band MISR.
        Returns list of (lr_stack, hr, quality_mask) where:
        - lr_stack: [N, 1, 128, 128] N LR images, single band
        - hr: [1, 384, 384] HR reference
        - quality_mask: [384, 384] HR status mask
        """
        pairs = []

        for scene_id in scenes:
            lr_stack, hr_ref, split = self.get_scene_stack(scenes, scene_id)
            if hr_ref is None:
                continue  # Skip test scenes without HR

            band_scene = next(iter(scenes[scene_id].values()))
            quality_mask = band_scene.hr_status_mask

            # Stack LR into [N, 1, 128, 128]
            lr_array = np.stack(lr_stack, axis=0)  # [N, 1, 128, 128]

            pairs.append((lr_array, hr_ref, quality_mask))

        return pairs

    def split_train_val(self, train_pairs: List, val_fraction: float = 0.1,
                        seed: int = 42) -> Tuple[List, List]:
        """Split training pairs into train/val."""
        np.random.seed(seed)
        indices = np.random.permutation(len(train_pairs))
        n_val = int(len(train_pairs) * val_fraction)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_pairs = [train_pairs[i] for i in val_indices]
        train_pairs = [train_pairs[i] for i in train_indices]

        return train_pairs, val_pairs


def main():
    logging.basicConfig(level=logging.INFO)
    loader = ProbaVLoader()

    # Check data exists
    for split in ['train', 'test']:
        for band in loader.bands:
            scenes = loader.list_scenes(split, band)
            logger.info(f"{split}/{band}: {len(scenes)} scenes")

    # Load train scenes
    train_scenes = loader.load_all_scenes('train')
    logger.info(f"Loaded {len(train_scenes)} training scenes")

    # Prepare pairs
    pairs = loader.prepare_training_pairs(train_scenes)
    logger.info(f"Prepared {len(pairs)} training pairs")

    # Split
    train_pairs, val_pairs = loader.split_train_val(pairs, val_fraction=0.1)
    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Check shapes
    if train_pairs:
        lr, hr, mask = train_pairs[0]
        logger.info(f"LR stack shape: {lr.shape}")  # [N, 2, 128, 128]
        logger.info(f"HR shape: {hr.shape}")  # [2, 384, 384]
        logger.info(f"Mask shape: {mask.shape}")  # [384, 384]


if __name__ == "__main__":
    main()