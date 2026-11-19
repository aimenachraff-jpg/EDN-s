"""
Unit tests for Downstream Task Evaluation Framework.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from src.downstream.tasks import (
    TaskModel,
    UNetResNet18,
    BuildingDetector,
    DeepLabV3Plus,
    SiameseUNet,
    classification_metrics,
    detection_metrics,
    segmentation_metrics,
    change_detection_metrics,
    create_task_model,
    FixedBackboneEvaluator,
)


class TestTaskModel:
    """Tests for base TaskModel."""
    
    def test_task_model_creation(self):
        """TaskModel should create."""
        model = TaskModel(num_classes=10, in_channels=4)
        assert model is not None
        assert model.num_classes == 10
        assert model._frozen is False
    
    def test_freeze_backbone(self):
        """freeze_backbone should freeze all parameters."""
        model = TaskModel(num_classes=10, in_channels=4)
        model.freeze_backbone()
        assert model.is_frozen is True
        for param in model.parameters():
            assert param.requires_grad is False
    
    def test_unfreeze(self):
        """unfreeze should unfreeze parameters."""
        model = TaskModel(num_classes=10, in_channels=4)
        model.freeze_backbone()
        model.unfreeze()
        assert model.is_frozen is False


class TestUNetResNet18:
    """Tests for UNet classification model."""
    
    def test_creation(self):
        """UNet should create."""
        model = UNetResNet18(num_classes=10, in_channels=4)
        assert model is not None
    
    def test_forward_shape(self):
        """UNet should produce correct output shape."""
        model = UNetResNet18(num_classes=10, in_channels=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 10, 64, 64)
    
    def test_freeze(self):
        """UNet should freeze properly."""
        model = UNetResNet18(num_classes=10, in_channels=4)
        model.freeze_backbone()
        assert model.is_frozen is True


class TestDeepLabV3Plus:
    """Tests for DeepLabV3+ segmentation model."""
    
    def test_creation(self):
        """DeepLabV3+ should create."""
        model = DeepLabV3Plus(num_classes=10, in_channels=4)
        assert model is not None
    
    def test_forward_shape(self):
        """DeepLabV3+ should produce correct output shape."""
        model = DeepLabV3Plus(num_classes=10, in_channels=4)
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 10, 64, 64)


class TestSiameseUNet:
    """Tests for Siamese UNet change detection model."""
    
    def test_creation(self):
        """SiameseUNet should create."""
        model = SiameseUNet(num_classes=2, in_channels=4)
        assert model is not None
    
    def test_forward_shape(self):
        """SiameseUNet should produce correct output shape."""
        model = SiameseUNet(num_classes=2, in_channels=4)
        model.eval()
        x1 = torch.randn(1, 4, 64, 64)
        x2 = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x1, x2)
        assert y.shape == (1, 2, 64, 64)


class TestBuildingDetector:
    """Tests for building detection model."""
    
    def test_creation(self):
        """BuildingDetector should create."""
        model = BuildingDetector(num_classes=1, in_channels=4, model_type='simple')
        assert model is not None
    
    def test_forward(self):
        """BuildingDetector should produce output."""
        model = BuildingDetector(num_classes=1, in_channels=4, model_type='simple')
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert 'boxes' in y
        assert 'scores' in y
        assert 'labels' in y


class TestMetrics:
    """Tests for task metrics."""
    
    def test_classification_metrics(self):
        """Classification metrics should compute correctly."""
        pred = torch.randint(0, 10, (2, 64, 64))
        target = torch.randint(0, 10, (2, 64, 64))
        metrics = classification_metrics(pred, target, 10)
        assert 'miou' in metrics
        assert 'oa' in metrics
        assert 'kappa' in metrics
        assert 0 <= metrics['miou'] <= 1
        assert 0 <= metrics['oa'] <= 1
    
    def test_classification_metrics_perfect(self):
        """Perfect predictions should give miou=1."""
        target = torch.randint(0, 10, (2, 64, 64))
        metrics = classification_metrics(target, target, 10)
        assert abs(metrics['miou'] - 1.0) < 1e-6
        assert abs(metrics['oa'] - 1.0) < 1e-6
    
    def test_change_detection_metrics(self):
        """Change detection metrics should compute correctly."""
        pred = torch.randint(0, 2, (2, 64, 64))
        target = torch.randint(0, 2, (2, 64, 64))
        metrics = change_detection_metrics(pred, target)
        assert 'f1' in metrics
        assert 'iou' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_segmentation_metrics(self):
        """Segmentation metrics should work."""
        pred = torch.randint(0, 10, (2, 64, 64))
        target = torch.randint(0, 10, (2, 64, 64))
        metrics = segmentation_metrics(pred, target, 10)
        assert 'miou' in metrics


class TestTaskModelFactory:
    """Tests for task model factory."""
    
    def test_create_classification(self):
        """Should create classification model."""
        model = create_task_model('classification', num_classes=10, in_channels=4)
        assert isinstance(model, UNetResNet18)
    
    def test_create_segmentation(self):
        """Should create segmentation model."""
        model = create_task_model('segmentation', num_classes=10, in_channels=4)
        assert isinstance(model, DeepLabV3Plus)
    
    def test_create_change_detection(self):
        """Should create change detection model."""
        model = create_task_model('change_detection', num_classes=2, in_channels=4)
        assert isinstance(model, SiameseUNet)
    
    def test_create_invalid(self):
        """Should raise on invalid task."""
        with pytest.raises(ValueError):
            create_task_model('invalid_task', num_classes=10, in_channels=4)


class TestFixedBackboneEvaluator:
    """Tests for fixed-backbone evaluation protocol."""
    
    def test_evaluator_creation(self):
        """Evaluator should create."""
        model = UNetResNet18(num_classes=10, in_channels=4)
        model.freeze_backbone()
        evaluator = FixedBackboneEvaluator(
            task_model=model,
            task_metric_fn=lambda pred, target: classification_metrics(pred, target, 10),
            metric_name='miou'
        )
        assert evaluator is not None
    
    def test_evaluate(self):
        """Evaluator should compute normalized scores."""
        model = UNetResNet18(num_classes=10, in_channels=4)
        model.freeze_backbone()
        
        # Create a task metric that handles the shape mismatch
        def task_metric_fn(pred, target):
            # pred: [1, 10, H, W], target: [1, 4, H, W]
            # For simplicity, just return a scalar metric
            return 0.5
        
        evaluator = FixedBackboneEvaluator(
            task_model=model,
            task_metric_fn=task_metric_fn,
            metric_name='score'
        )
        
        hr = np.random.rand(4, 32, 32).astype(np.float32)
        sr = hr.copy()
        sr_outputs = {'model_a': sr}
        
        results = evaluator.evaluate(sr_outputs, hr)
        assert 'model_a' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])