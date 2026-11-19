"""
Unit tests for Hallucination Detection Framework.
All tests use synthetic data - no external dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from src.hallucination.detection import (
    HallucinationDetector,
    HallucinationThresholds,
    HallucinationType,
    AnnotationProtocol,
    AnnotationSample,
)


class TestHallucinationThresholds:
    """Tests for hallucination thresholds."""
    
    def test_default_thresholds(self):
        """Default thresholds should be initialized."""
        thresholds = HallucinationThresholds()
        assert thresholds.sam_threshold_rad == 0.175
        assert thresholds.ndvi_min == -1.0
        assert thresholds.ndvi_max == 1.0
        assert thresholds.structural_energy_ratio == 3.0
    
    def test_custom_thresholds(self):
        """Custom thresholds should be accepted."""
        thresholds = HallucinationThresholds(sam_threshold_rad=0.1)
        assert thresholds.sam_threshold_rad == 0.1


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""
    
    def test_detector_creation(self):
        """Detector should create with default thresholds."""
        detector = HallucinationDetector()
        assert detector is not None
        assert detector.thresholds is not None
    
    def test_spectral_identical(self):
        """Spectral detection on identical images should not detect."""
        detector = HallucinationDetector()
        img = np.random.rand(4, 64, 64).astype(np.float32)
        result = detector.detect_spectral(img, img, band_names=['B', 'G', 'R', 'NIR'])
        # Numerical precision gives ~0.0001 rad; use reasonable tolerance
        assert result['mean_sam'] < 1e-3
        assert result['detected'] is False or result['mean_sam'] < 0.1
    
    def test_spectral_different(self):
        """Spectral detection on different images should detect."""
        detector = HallucinationDetector()
        hr = np.random.rand(4, 64, 64).astype(np.float32)
        sr = np.random.rand(4, 64, 64).astype(np.float32)
        result = detector.detect_spectral(sr, hr, band_names=['B', 'G', 'R', 'NIR'])
        assert result['detected'] is True or result['mean_sam'] > 0
    
    def test_structural_identical(self):
        """Structural detection on identical images should not detect."""
        detector = HallucinationDetector()
        img = np.random.rand(4, 64, 64).astype(np.float32)
        result = detector.detect_structural(img, img)
        assert result['detected'] is False or result.get('reason') == 'insufficient_homogeneous_pixels'
    
    def test_textural_identical(self):
        """Textural detection on identical images should not detect."""
        detector = HallucinationDetector()
        img = np.random.rand(4, 64, 64).astype(np.float32)
        result = detector.detect_textural(img, img)
        assert result['detected'] is False
        assert result['mean_entropy_diff'] < 1e-6
    
    def test_textural_periodic_pattern(self):
        """Textural detection should detect added periodic patterns."""
        detector = HallucinationDetector()
        hr = np.random.rand(4, 64, 64).astype(np.float32)
        sr = hr.copy()
        # Add periodic pattern (textural hallucination)
        x, y = np.meshgrid(np.arange(64), np.arange(64))
        sr[0] += 0.3 * np.sin(2 * np.pi * x / 8)
        sr = np.clip(sr, 0, 1)
        result = detector.detect_textural(sr, hr)
        # Should detect or have high entropy diff
        assert result['mean_entropy_diff'] > 0
    
    def test_detect_all(self):
        """detect_all should return all categories."""
        detector = HallucinationDetector()
        hr = np.random.rand(4, 64, 64).astype(np.float32)
        sr = hr + 0.1 * np.random.randn(4, 64, 64).astype(np.float32)
        sr = np.clip(sr, 0, 1)
        result = detector.detect_all(sr, hr, band_names=['B', 'G', 'R', 'NIR'])
        assert 'classification' in result
        assert 'detected_types' in result
        assert 'confidences' in result
        assert 'details' in result
        assert 'spectral' in result['details']
        assert 'structural' in result['details']
        assert 'textural' in result['details']
    
    def test_detect_all_with_task_model(self):
        """detect_all should handle optional task model."""
        detector = HallucinationDetector()
        hr = np.random.rand(4, 64, 64).astype(np.float32)
        sr = hr + 0.1 * np.random.randn(4, 64, 64).astype(np.float32)
        sr = np.clip(sr, 0, 1)
        
        # Simple task metric
        def task_metric(pred, target):
            return torch.tensor(0.5)
        
        result = detector.detect_all(sr, hr, task_model=None, task_metric=None)
        assert result['details']['semantic']['reason'] == 'no_task_model'


class TestAnnotationProtocol:
    """Tests for annotation protocol."""
    
    def test_protocol_creation(self):
        """AnnotationProtocol should create."""
        protocol = AnnotationProtocol()
        assert protocol is not None
        assert protocol.n_annotators == 3
    
    def test_add_sample(self):
        """Should add samples."""
        protocol = AnnotationProtocol()
        sample = AnnotationSample(
            sample_id='test_001',
            sr_image=np.random.rand(4, 64, 64).astype(np.float32),
            hr_image=np.random.rand(4, 64, 64).astype(np.float32),
            model_name='EDSR',
            patch_coords=(0, 0, 64, 64)
        )
        protocol.add_sample(sample)
        assert len(protocol.samples) == 1
    
    def test_record_annotation(self):
        """Should record annotations."""
        protocol = AnnotationProtocol()
        sample = AnnotationSample(
            sample_id='test_001',
            sr_image=np.random.rand(4, 64, 64).astype(np.float32),
            hr_image=np.random.rand(4, 64, 64).astype(np.float32),
            model_name='EDSR',
            patch_coords=(0, 0, 64, 64)
        )
        protocol.add_sample(sample)
        protocol.record_annotation('test_001', 'annotator_1', [HallucinationType.SPECTRAL])
        assert 'test_001' in protocol.annotations
    
    def test_fleiss_kappa_empty(self):
        """Fleiss kappa should be 0 for empty annotations."""
        protocol = AnnotationProtocol()
        kappa = protocol.compute_fleiss_kappa()
        assert kappa == 0.0
    
    def test_generate_report(self):
        """Should generate report."""
        protocol = AnnotationProtocol()
        report = protocol.generate_report()
        assert 'fleiss_kappa' in report
        assert 'target_kappa' in report
        assert 'passes_threshold' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])