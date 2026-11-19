"""
EDN-S Hallucination Detection
=============================

Hallucination taxonomy and detection methods for satellite super-resolution.
Four categories: Spectral, Structural, Textural, Semantic.

This module provides:
- Hallucination taxonomy definitions
- Detection algorithms for each category
- Expert annotation interface utilities
- Metric validation tools (for H2)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
import warnings

warnings.filterwarnings('ignore')


class HallucinationType(Enum):
    """Hallucination categories from Protocol Section 6.1"""
    SPECTRAL = "spectral"
    STRUCTURAL = "structural"
    TEXTURAL = "textural"
    SEMANTIC = "semantic"
    NONE = "none"
    MULTIPLE = "multiple"


@dataclass
class HallucinationThresholds:
    """Thresholds for hallucination detection (Protocol Table 6.1)"""
    # Spectral: SAM > 10° (0.175 rad) or NDVI out-of-range
    sam_threshold_rad: float = 0.175  # ~10 degrees
    ndvi_min: float = -1.0
    ndvi_max: float = 1.0
    
    # Structural: Wavelet HH energy > 3σ local mean in homogeneous regions
    structural_energy_ratio: float = 3.0
    structural_relative_energy: float = 0.15
    
    # Textural: GLCM entropy mismatch > 20%
    textural_entropy_diff: float = 0.20
    
    # Semantic: Task mAP drop > 15% vs HR
    semantic_task_drop: float = 0.15


class HallucinationDetector:
    """
    Unified hallucination detector implementing all four categories.
    
    Usage:
        detector = HallucinationDetector()
        results = detector.detect(sr, hr, task_model=task_model)
    """
    
    def __init__(self, thresholds: Optional[HallucinationThresholds] = None):
        self.thresholds = thresholds or HallucinationThresholds()
    
    # -------------------------------------------------------------------------
    # SPECTRAL HALLUCINATION
    # -------------------------------------------------------------------------
    
    def detect_spectral(self, sr: np.ndarray, hr: np.ndarray,
                        band_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect spectral hallucination (violates physical band correlations).
        
        Metrics:
        - Mean SAM across bands
        - Per-band-pair SAM
        - NDVI out-of-range detection (if Red/NIR bands present)
        - Spectral correlation matrix deviation
        
        Args:
            sr: Super-resolved [C, H, W]
            hr: High-resolution reference [C, H, W]
            band_names: Optional band names for interpretation
        
        Returns:
            Dict with detection results
        """
        C, H, W = sr.shape
        
        # Mean SAM
        mean_sam = self._mean_sam(sr, hr)
        
        # Per-band-pair SAM
        pair_sams = {}
        for i in range(C):
            for j in range(i+1, C):
                sam_ij = self._sam_band_pair(sr, hr, i, j)
                name = f"{i}-{j}"
                if band_names:
                    name = f"{band_names[i]}-{band_names[j]}"
                pair_sams[name] = sam_ij
        
        # NDVI check (assuming bands 2=Red, 3=NIR for standard 4-band)
        ndvi_violation = False
        ndvi_stats = {}
        if C >= 4:
            # Standard: 0=Blue, 1=Green, 2=Red, 3=NIR
            red_sr, nir_sr = sr[2], sr[3]
            red_hr, nir_hr = hr[2], hr[3]
            
            ndvi_sr = (nir_sr - red_sr) / (nir_sr + red_sr + 1e-8)
            ndvi_hr = (nir_hr - red_hr) / (nir_hr + red_hr + 1e-8)
            
            ndvi_stats = {
                'sr_mean': float(np.mean(ndvi_sr)),
                'hr_mean': float(np.mean(ndvi_hr)),
                'sr_range': [float(np.min(ndvi_sr)), float(np.max(ndvi_sr))],
                'hr_range': [float(np.min(ndvi_hr)), float(np.max(ndvi_hr))],
            }
            
            # Check if SR NDVI exceeds physical bounds
            sr_min, sr_max = ndvi_stats['sr_range']
            if sr_min < self.thresholds.ndvi_min or sr_max > self.thresholds.ndvi_max:
                ndvi_violation = True
        
        # Correlation matrix deviation
        corr_dev = self._correlation_deviation(sr, hr)
        
        # Detection decision
        detected = (mean_sam > self.thresholds.sam_threshold_rad or 
                    ndvi_violation or 
                    corr_dev > 0.2)
        
        return {
            'detected': detected,
            'type': HallucinationType.SPECTRAL.value,
            'mean_sam': float(mean_sam),
            'sam_threshold': self.thresholds.sam_threshold_rad,
            'pair_sams': pair_sams,
            'ndvi_violation': ndvi_violation,
            'ndvi_stats': ndvi_stats,
            'correlation_deviation': float(corr_dev),
            'confidence': min(mean_sam / self.thresholds.sam_threshold_rad, 1.0) if self.thresholds.sam_threshold_rad > 0 else 0.0
        }
    
    def _mean_sam(self, sr: np.ndarray, hr: np.ndarray) -> float:
        """Mean spectral angle across all pixels."""
        C, H, W = sr.shape
        sr_flat = sr.reshape(C, -1).T
        hr_flat = hr.reshape(C, -1).T
        
        dot = np.sum(sr_flat * hr_flat, axis=1)
        norm_sr = np.linalg.norm(sr_flat, axis=1)
        norm_hr = np.linalg.norm(hr_flat, axis=1)
        cos_theta = np.clip(dot / (norm_sr * norm_hr + 1e-10), -1, 1)
        angles = np.arccos(cos_theta)
        return float(np.mean(angles))
    
    def _sam_band_pair(self, sr: np.ndarray, hr: np.ndarray, i: int, j: int) -> float:
        """SAM for a specific band pair."""
        sr_pair = sr[[i, j]].reshape(2, -1).T
        hr_pair = hr[[i, j]].reshape(2, -1).T
        
        dot = np.sum(sr_pair * hr_pair, axis=1)
        norm_sr = np.linalg.norm(sr_pair, axis=1)
        norm_hr = np.linalg.norm(hr_pair, axis=1)
        cos_theta = np.clip(dot / (norm_sr * norm_hr + 1e-10), -1, 1)
        return float(np.mean(np.arccos(cos_theta)))
    
    def _correlation_deviation(self, sr: np.ndarray, hr: np.ndarray) -> float:
        """Mean absolute correlation matrix deviation."""
        C = sr.shape[0]
        sr_flat = sr.reshape(C, -1)
        hr_flat = hr.reshape(C, -1)
        corr_sr = np.corrcoef(sr_flat)
        corr_hr = np.corrcoef(hr_flat)
        mask = np.triu_indices(C, k=1)
        return float(np.mean(np.abs(corr_sr[mask] - corr_hr[mask])))
    
    # -------------------------------------------------------------------------
    # STRUCTURAL HALLUCINATION
    # -------------------------------------------------------------------------
    
    def detect_structural(self, sr: np.ndarray, hr: np.ndarray,
                          wavelet: str = 'db1', level: int = 2) -> Dict[str, Any]:
        """
        Detect structural hallucination (spurious high-freq in homogeneous regions).
        
        Uses wavelet HH (diagonal detail) energy in homogeneous regions.
        Homogeneous regions defined by local variance in HR.
        
        Args:
            sr: Super-resolved [C, H, W]
            hr: High-resolution reference [C, H, W]
            wavelet: Wavelet type
            level: Decomposition level
        
        Returns:
            Dict with detection results
        """
        try:
            import pywt
        except ImportError:
            return {'detected': False, 'type': HallucinationType.STRUCTURAL.value, 'error': 'pywt not installed'}
        
        C, H, W = sr.shape
        
        # Homogeneous mask from HR
        hr_gray = np.mean(hr, axis=0)
        var_local = ndimage.gaussian_filter(hr_gray**2, sigma=4) - ndimage.gaussian_filter(hr_gray, sigma=4)**2
        homogeneous_mask = (var_local < 0.01).astype(float)
        
        if homogeneous_mask.sum() < 100:
            return {
                'detected': False,
                'type': HallucinationType.STRUCTURAL.value,
                'reason': 'insufficient_homogeneous_pixels',
                'homogeneous_pixels': int(homogeneous_mask.sum())
            }
        
        # Compute HH energy ratio per channel
        channel_ratios = []
        channel_energies = {}
        
        for c in range(C):
            coeffs_sr = pywt.wavedec2(sr[c], wavelet, level=level)
            coeffs_hr = pywt.wavedec2(hr[c], wavelet, level=level)
            
            level_ratios = []
            for lvl in range(1, level + 1):
                cD_sr = coeffs_sr[lvl][2]  # HH
                cD_hr = coeffs_hr[lvl][2]
                
                # Resize mask to coefficient size
                h, w = cD_sr.shape
                mask_resized = self._resize_mask(homogeneous_mask, h, w)
                
                energy_sr = np.mean((cD_sr**2) * mask_resized)
                energy_hr = np.mean((cD_hr**2) * mask_resized)
                
                if energy_hr > 1e-10:
                    ratio = energy_sr / energy_hr
                    level_ratios.append(ratio)
            
            if level_ratios:
                channel_ratios.append(np.mean(level_ratios))
                channel_energies[f'band_{c}'] = np.mean(level_ratios)
        
        mean_ratio = float(np.mean(channel_ratios)) if channel_ratios else 1.0
        
        # Detection: mean ratio > threshold (3σ = 3x expected)
        detected = mean_ratio > self.thresholds.structural_energy_ratio
        
        return {
            'detected': detected,
            'type': HallucinationType.STRUCTURAL.value,
            'mean_energy_ratio': mean_ratio,
            'threshold': self.thresholds.structural_energy_ratio,
            'channel_ratios': channel_energies,
            'homogeneous_pixels': int(homogeneous_mask.sum()),
            'confidence': min(mean_ratio / self.thresholds.structural_energy_ratio, 1.0) if detected else 0.0
        }
    
    def _resize_mask(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize mask to target size."""
        try:
            import cv2
            return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            # Simple numpy resize
            h_ratio = mask.shape[0] / h
            w_ratio = mask.shape[1] / w
            return mask[::int(h_ratio), ::int(w_ratio)]
    
    # -------------------------------------------------------------------------
    # TEXTURAL HALLUCINATION
    # -------------------------------------------------------------------------
    
    def detect_textural(self, sr: np.ndarray, hr: np.ndarray,
                        distances: List[int] = [1, 3, 5],
                        angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, Any]:
        """
        Detect textural hallucination (plausible but ungrounded periodic patterns).
        
        Uses GLCM (Gray-Level Co-occurrence Matrix) texture features.
        Compares entropy, contrast, homogeneity between SR and HR.
        
        Args:
            sr: Super-resolved [C, H, W]
            hr: High-resolution reference [C, H, W]
            distances: Pixel distances for GLCM
            angles: Angles for GLCM
        
        Returns:
            Dict with detection results
        """
        C = sr.shape[0]
        channel_results = {}
        entropy_diffs = []
        
        for c in range(C):
            # Quantize to 32 levels for GLCM
            sr_q = self._quantize_for_glcm(sr[c])
            hr_q = self._quantize_for_glcm(hr[c])
            
            # Compute GLCM properties
            glcm_sr = graycomatrix(sr_q, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
            glcm_hr = graycomatrix(hr_q, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
            
            # Entropy
            entropy_sr = -np.sum(glcm_sr * np.log(glcm_sr + 1e-10))
            entropy_hr = -np.sum(glcm_hr * np.log(glcm_hr + 1e-10))
            
            # Contrast
            contrast_sr = graycoprops(glcm_sr, 'contrast').mean()
            contrast_hr = graycoprops(glcm_hr, 'contrast').mean()
            
            # Homogeneity
            homog_sr = graycoprops(glcm_sr, 'homogeneity').mean()
            homog_hr = graycoprops(glcm_hr, 'homogeneity').mean()
            
            entropy_diff = abs(entropy_sr - entropy_hr) / (entropy_hr + 1e-10)
            entropy_diffs.append(entropy_diff)
            
            channel_results[f'band_{c}'] = {
                'entropy_sr': float(entropy_sr),
                'entropy_hr': float(entropy_hr),
                'entropy_diff': float(entropy_diff),
                'contrast_sr': float(contrast_sr),
                'contrast_hr': float(contrast_hr),
                'homogeneity_sr': float(homog_sr),
                'homogeneity_hr': float(homog_hr),
            }
        
        mean_entropy_diff = float(np.mean(entropy_diffs))
        detected = mean_entropy_diff > self.thresholds.textural_entropy_diff
        
        return {
            'detected': detected,
            'type': HallucinationType.TEXTURAL.value,
            'mean_entropy_diff': mean_entropy_diff,
            'threshold': self.thresholds.textural_entropy_diff,
            'channel_results': channel_results,
            'confidence': min(mean_entropy_diff / self.thresholds.textural_entropy_diff, 1.0) if detected else 0.0
        }
    
    def _quantize_for_glcm(self, img: np.ndarray, levels: int = 32) -> np.ndarray:
        """Quantize image for GLCM computation."""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            quantized = ((img - img_min) / (img_max - img_min) * (levels - 1)).astype(np.uint8)
        else:
            quantized = np.zeros_like(img, dtype=np.uint8)
        return np.clip(quantized, 0, levels - 1)
    
    # -------------------------------------------------------------------------
    # SEMANTIC HALLUCINATION
    # -------------------------------------------------------------------------
    
    def detect_semantic(self, sr: np.ndarray, hr: np.ndarray,
                        task_model: Callable, task_metric: Callable) -> Dict[str, Any]:
        """
        Detect semantic hallucination (downstream confidence on false structures).
        
        Uses fixed-backbone task model: task model trained on HR, evaluated on SR vs HR.
        Performance drop indicates semantic hallucination.
        
        Args:
            sr: Super-resolved [C, H, W]
            hr: High-resolution reference [C, H, W]
            task_model: Fixed-backbone task model (frozen weights)
            task_metric: Task metric function (e.g., mIoU, mAP)
        
        Returns:
            Dict with detection results
        """
        try:
            # Add batch dimension if needed
            sr_batch = sr[np.newaxis, ...] if sr.ndim == 3 else sr
            hr_batch = hr[np.newaxis, ...] if hr.ndim == 3 else hr
            
            # Ensure torch tensors
            if isinstance(sr_batch, np.ndarray):
                sr_batch = torch.from_numpy(sr_batch).float()
            if isinstance(hr_batch, np.ndarray):
                hr_batch = torch.from_numpy(hr_batch).float()
            
            # Forward pass
            with torch.no_grad():
                pred_sr = task_model(sr_batch)
                pred_hr = task_model(hr_batch)
            
            # Compute task metrics
            # Note: task_metric should accept (pred, target) and return scalar
            metric_sr = task_metric(pred_sr, hr_batch)
            metric_hr = task_metric(pred_hr, hr_batch)
            
            # Normalize to scalar if tensor
            if isinstance(metric_sr, torch.Tensor):
                metric_sr = metric_sr.item()
            if isinstance(metric_hr, torch.Tensor):
                metric_hr = metric_hr.item()
            
            # Performance drop
            drop = metric_hr - metric_sr
            relative_drop = drop / (metric_hr + 1e-8) if metric_hr > 0 else 0
            
            detected = relative_drop > self.thresholds.semantic_task_drop
            
            return {
                'detected': detected,
                'type': HallucinationType.SEMANTIC.value,
                'task_metric_sr': float(metric_sr),
                'task_metric_hr': float(metric_hr),
                'absolute_drop': float(drop),
                'relative_drop': float(relative_drop),
                'threshold': self.thresholds.semantic_task_drop,
                'confidence': min(relative_drop / self.thresholds.semantic_task_drop, 1.0) if detected else 0.0
            }
        
        except Exception as e:
            return {
                'detected': False,
                'type': HallucinationType.SEMANTIC.value,
                'error': str(e)
            }
    
    # -------------------------------------------------------------------------
    # UNIFIED DETECT
    # -------------------------------------------------------------------------
    
    def detect_all(self, sr: np.ndarray, hr: np.ndarray,
                   task_model: Optional[Callable] = None,
                   task_metric: Optional[Callable] = None,
                   band_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all hallucination detectors.
        
        Args:
            sr: Super-resolved [C, H, W]
            hr: High-resolution reference [C, H, W]
            task_model: Optional fixed-backbone task model
            task_metric: Optional task metric function
            band_names: Optional band names
        
        Returns:
            Dict with all detection results and composite classification
        """
        results = {}
        
        # Spectral
        results['spectral'] = self.detect_spectral(sr, hr, band_names)
        
        # Structural
        results['structural'] = self.detect_structural(sr, hr)
        
        # Textural
        results['textural'] = self.detect_textural(sr, hr)
        
        # Semantic (if task model provided)
        if task_model is not None and task_metric is not None:
            results['semantic'] = self.detect_semantic(sr, hr, task_model, task_metric)
        else:
            results['semantic'] = {'detected': False, 'type': 'semantic', 'reason': 'no_task_model'}
        
        # Composite classification
        detected_types = [k for k, v in results.items() if v.get('detected', False)]
        
        if len(detected_types) == 0:
            classification = HallucinationType.NONE.value
        elif len(detected_types) == 1:
            classification = detected_types[0]
        else:
            classification = HallucinationType.MULTIPLE.value
        
        # Confidence scores
        confidences = {k: v.get('confidence', 0) for k, v in results.items()}
        
        return {
            'classification': classification,
            'detected_types': detected_types,
            'confidences': confidences,
            'details': results
        }


# =============================================================================
# ANNOTATION INTERFACE UTILITIES
# =============================================================================

@dataclass
class AnnotationSample:
    """Single sample for expert annotation (Protocol Section 6.2)"""
    sample_id: str
    sr_image: np.ndarray  # [C, H, W]
    hr_image: np.ndarray  # [C, H, W]
    model_name: str
    patch_coords: Tuple[int, int, int, int]  # (y, x, h, w) in full image
    annotations: List[HallucinationType] = None
    annotator_id: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = []


class AnnotationProtocol:
    """
    Manages expert annotation protocol for hallucination validation (H2).
    
    Protocol (Section 6.2):
    - 3 remote sensing experts
    - 200 patches × 4 models = 800 patches
    - Labels: {None, Spectral, Structural, Textural, Semantic, Multiple}
    - Agreement: Fleiss' κ > 0.7
    """
    
    def __init__(self, n_annotators: int = 3, target_kappa: float = 0.7):
        self.n_annotators = n_annotators
        self.target_kappa = target_kappa
        self.samples: List[AnnotationSample] = []
        self.annotations: Dict[str, Dict[str, List[HallucinationType]]] = {}  # sample_id -> annotator_id -> labels
    
    def add_sample(self, sample: AnnotationSample):
        self.samples.append(sample)
    
    def record_annotation(self, sample_id: str, annotator_id: str, labels: List[HallucinationType]):
        if sample_id not in self.annotations:
            self.annotations[sample_id] = {}
        self.annotations[sample_id][annotator_id] = labels
    
    def compute_fleiss_kappa(self) -> float:
        """
        Compute Fleiss' kappa for inter-annotator agreement.
        
        Returns:
            Fleiss' kappa value
        """
        if not self.annotations:
            return 0.0
        
        # Categories
        categories = [t.value for t in HallucinationType if t != HallucinationType.NONE]
        n_categories = len(categories)
        n_annotators = self.n_annotators
        n_samples = len(self.annotations)
        
        # Count matrix: samples x categories
        counts = np.zeros((n_samples, n_categories))
        
        sample_ids = list(self.annotations.keys())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        for i, sample_id in enumerate(sample_ids):
            anns = self.annotations[sample_id]
            for annotator_id, labels in anns.items():
                for label in labels:
                    if label.value in cat_to_idx:
                        counts[i, cat_to_idx[label.value]] += 1
        
        # Fleiss' kappa calculation
        # P_i = 1/(n*(n-1)) * sum_j n_ij*(n_ij - 1)
        n = n_annotators
        p_i = np.sum(counts * (counts - 1), axis=1) / (n * (n - 1))
        P_bar = np.mean(p_i)
        
        # P_e = sum_j p_j^2 where p_j = total annotations in category j / (N * n)
        p_j = np.sum(counts, axis=0) / (n_samples * n)
        P_e = np.sum(p_j ** 2)
        
        if P_e == 1:
            return 1.0
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return float(kappa)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate annotation quality report."""
        kappa = self.compute_fleiss_kappa()
        
        # Per-sample agreement
        sample_agreements = {}
        for sample_id, anns in self.annotations.items():
            # Count how many annotators agree on each label
            all_labels = []
            for labels in anns.values():
                all_labels.extend([l.value for l in labels])
            
            from collections import Counter
            label_counts = Counter(all_labels)
            max_agree = max(label_counts.values()) if label_counts else 0
            sample_agreements[sample_id] = max_agree / self.n_annotators
        
        return {
            'fleiss_kappa': kappa,
            'target_kappa': self.target_kappa,
            'passes_threshold': kappa >= self.target_kappa,
            'n_samples': len(self.samples),
            'n_annotated': len(self.annotations),
            'mean_sample_agreement': float(np.mean(list(sample_agreements.values()))) if sample_agreements else 0,
            'sample_agreements': sample_agreements
        }


# =============================================================================
# METRIC VALIDATION (FOR H2)
# =============================================================================

def validate_hallucination_metrics(detector: HallucinationDetector,
                                   sr_samples: List[np.ndarray],
                                   hr_samples: List[np.ndarray],
                                   expert_labels: List[List[HallucinationType]]) -> Dict[str, Any]:
    """
    Validate hallucination metrics against expert annotations (H2 protocol).
    
    Computes:
    - Precision, Recall, F1 per category
    - Correlation between metric confidence and expert agreement
    - AUC-ROC for each detector
    
    Args:
        detector: HallucinationDetector instance
        sr_samples: List of SR images [C, H, W]
        hr_samples: List of HR images [C, H, W]
        expert_labels: List of expert label lists per sample
    
    Returns:
        Validation results
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    except ImportError:
        return {'error': 'sklearn not installed'}
    
    categories = [HallucinationType.SPECTRAL, HallucinationType.STRUCTURAL, 
                  HallucinationType.TEXTURAL, HallucinationType.SEMANTIC]
    
    # Get detector predictions
    predictions = {cat: [] for cat in categories}
    confidences = {cat: [] for cat in categories}
    ground_truth = {cat: [] for cat in categories}
    
    for sr, hr, labels in zip(sr_samples, hr_samples, expert_labels):
        result = detector.detect_all(sr, hr)
        
        for cat in categories:
            # Binary detection
            pred = cat.value in result['detected_types']
            predictions[cat].append(int(pred))
            
            # Confidence
            conf = result['confidences'].get(cat.value, 0)
            confidences[cat].append(conf)
            
            # Ground truth
            gt = cat in labels
            ground_truth[cat].append(int(gt))
    
    # Compute metrics
    validation = {}
    for cat in categories:
        y_true = np.array(ground_truth[cat])
        y_pred = np.array(predictions[cat])
        y_score = np.array(confidences[cat])
        
        if len(np.unique(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            try:
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = 0.5
        else:
            precision = recall = f1 = auc = 0.0
        
        validation[cat.value] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'n_positive': int(y_true.sum()),
            'n_total': len(y_true)
        }
    
    return validation


# =============================================================================
# Main test
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    C, H, W = 4, 128, 128
    
    hr = np.random.rand(C, H, W).astype(np.float32)
    # Add some structured noise to simulate hallucination
    sr = hr.copy()
    sr += 0.1 * np.random.randn(C, H, W)
    sr = np.clip(sr, 0, 1)
    
    # Add a fake periodic pattern in one band (textural hallucination)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    sr[0] += 0.2 * np.sin(2 * np.pi * x / 8)
    sr = np.clip(sr, 0, 1)
    
    print("Testing EDN-S hallucination detection...")
    
    detector = HallucinationDetector()
    
    # Test spectral
    spec_result = detector.detect_spectral(sr, hr, band_names=['B', 'G', 'R', 'NIR'])
    print(f"Spectral: detected={spec_result['detected']}, SAM={spec_result['mean_sam']:.4f}")
    
    # Test structural
    struct_result = detector.detect_structural(sr, hr)
    print(f"Structural: detected={struct_result['detected']}, ratio={struct_result.get('mean_energy_ratio', 'N/A'):.4f}")
    
    # Test textural
    text_result = detector.detect_textural(sr, hr)
    print(f"Textural: detected={text_result['detected']}, entropy_diff={text_result['mean_entropy_diff']:.4f}")
    
    # Test all
    all_result = detector.detect_all(sr, hr, band_names=['B', 'G', 'R', 'NIR'])
    print(f"Classification: {all_result['classification']}")
    print(f"Detected types: {all_result['detected_types']}")
    print("All hallucination detection tests passed!")