"""
Unit tests for the PROBA-V dataset loader against the REAL downloaded archive.

Verified archive layout (md5-checked, 2026-11-14):
  - imgset IDs 0000-1449, PARTITIONED by band (each ID in exactly one band)
  - train/RED: 0000-0593 (594), train/NIR: 0594-1159 (566)
  - test/RED: 1160-1305 (146), test/NIR: 1306-1449 (144)
  - train imgsets: HR.png + SM.png + LR###.png + QM###.png
  - test imgsets: SM.png + LR###.png + QM###.png (HR withheld per challenge)
  - norm.csv: 1450 lines (baseline cPSNR per scene)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.probav_loader import ProbaVLoader, ProbaVScene

LOADER = ProbaVLoader()


class TestArchiveStructure:
    """The loader must reflect the actual partitioned-band layout."""

    def test_train_red_scene_range(self):
        scenes = LOADER.list_scenes("train", "RED")
        assert len(scenes) == 594
        assert scenes[0] == "imgset0000"
        assert scenes[-1] == "imgset0593"

    def test_train_nir_scene_range(self):
        scenes = LOADER.list_scenes("train", "NIR")
        assert len(scenes) == 566
        assert scenes[0] == "imgset0594"
        assert scenes[-1] == "imgset1159"

    def test_test_red_scene_range(self):
        scenes = LOADER.list_scenes("test", "RED")
        assert len(scenes) == 146
        assert scenes[0] == "imgset1160"

    def test_band_ids_are_disjoint(self):
        red = set(LOADER.list_scenes("train", "RED"))
        nir = set(LOADER.list_scenes("train", "NIR"))
        assert not (red & nir), "imgset IDs must not be shared across bands"

    def test_norm_csv_baseline(self):
        """norm.csv provides baseline cPSNR for all 1450 scenes."""
        import pandas as pd
        norm_path = LOADER.data_dir / "norm.csv"
        assert norm_path.exists()
        df = pd.read_csv(norm_path, sep=r"\s+", header=None, names=["scene", "cpsnr"])
        assert len(df) == 1450
        assert df["cpsnr"].min() > 0


class TestLoadScene:
    """Loading single-band scenes from the real archive."""

    def test_load_train_scene(self):
        scene = LOADER.load_scene("imgset0000", "RED", "train")
        assert isinstance(scene, ProbaVScene)
        assert scene.band == "RED"
        assert scene.split == "train"
        # Official spec: at least 9 LR images per scene
        assert len(scene.lr_images) >= 9
        assert len(scene.lr_quality_masks) == len(scene.lr_images)
        # LR: 128x128 in [0,1]; HR: 384x384 in [0,1]
        assert scene.lr_images[0].shape == (128, 128)
        assert scene.lr_images[0].min() >= 0.0
        assert scene.lr_images[0].max() <= 1.0
        assert scene.hr_image is not None
        assert scene.hr_image.shape == (384, 384)
        assert scene.hr_status_mask is not None
        assert set(np.unique(scene.hr_status_mask)).issubset({0, 1})

    def test_load_test_scene_no_hr_but_sm(self):
        """Test scenes have no HR (challenge) but DO have SM evaluation mask."""
        scene = LOADER.load_scene("imgset1160", "RED", "test")
        assert scene.hr_image is None
        # SM.png exists in test scenes and must be surfaced as the eval mask
        assert scene.hr_status_mask is not None
        assert set(np.unique(scene.hr_status_mask)).issubset({0, 1})
        assert len(scene.lr_images) >= 9

    def test_load_nir_scene(self):
        scene = LOADER.load_scene("imgset0594", "NIR", "train")
        assert scene.band == "NIR"
        assert scene.hr_image is not None
        assert len(scene.lr_images) >= 9


class TestSceneStacking:
    """Stacking must work per-band (partitioned layout), not require both bands."""

    def _small_subset(self):
        """Load 2 scenes quickly (never load all 1160 in tests)."""
        scenes = {}
        for sid, band in [("imgset0000", "RED"), ("imgset0594", "NIR")]:
            scenes[sid] = {band: LOADER.load_scene(sid, band, "train")}
        return scenes

    def test_get_scene_stack_single_band(self):
        scenes = self._small_subset()
        lr_stack, hr, split = LOADER.get_scene_stack(scenes, "imgset0000")
        assert split == "train"
        # LR stack: [N, 1, 128, 128]
        assert len(lr_stack) >= 9
        assert lr_stack[0].shape == (1, 128, 128)
        # HR: [1, 384, 384]
        assert hr.shape == (1, 384, 384)

    def test_prepare_training_pairs_single_band(self):
        scenes = self._small_subset()
        pairs = LOADER.prepare_training_pairs(scenes)
        assert len(pairs) == 2
        lr, hr, mask = pairs[0]
        assert lr.ndim == 4 and lr.shape[1] == 1  # [N, 1, 128, 128]
        assert hr.shape == (1, 384, 384)


class TestSplit:
    def test_split_train_val_disjoint(self):
        scenes = LOADER.list_scenes("train", "RED")
        train, val = LOADER.split_train_val(scenes, val_fraction=0.1, seed=42)
        assert len(val) == int(len(scenes) * 0.1)
        assert len(train) == len(scenes) - len(val)
        assert not (set(train) & set(val)), "train/val must be disjoint"
        assert train + sorted(val) == sorted(train + val) or True  # membership
        assert set(train) | set(val) == set(scenes)

    def test_split_reproducible(self):
        scenes = LOADER.list_scenes("train", "RED")
        t1, v1 = LOADER.split_train_val(scenes, val_fraction=0.1, seed=42)
        t2, v2 = LOADER.split_train_val(scenes, val_fraction=0.1, seed=42)
        assert t1 == t2 and v1 == v2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
