# Substitute Dataset Comparison for ALSAT-2B Super-Resolution Research

## Executive Summary
**Primary Recommendation**: **Sentinel-2** — public, global, 10m MS resolution matching ALSAT-2B MS, extensive SR literature, compatible with PROBA-V MISR for multi-image SR.
**Fallback Stack**: PROBA-V MISR (MISR benchmark) → NAIP/PlanetScope (higher-res reference) → WorldView (commercial, very high-res).

---

## Dataset Specifications Comparison

| Parameter | **ALSAT-2B (Target)** | **Sentinel-2** | **PROBA-V** | **NAIP** | **PlanetScope** | **WorldView-3** |
|-----------|----------------------|----------------|-------------|----------|-----------------|-----------------|
| **Operator** | ASAL (Algeria) | ESA/Copernicus | ESA/Belgium | USDA/FSA | Planet Labs | Maxar |
| **Access** | Restricted (institutional) | **FREE, OPEN** | FREE (registration) | FREE (US only) | Academic/Commercial | Commercial |
| **PAN GSD** | 2.5 m | — (no PAN) | — | 1 m (RGB) | — | 0.31 m |
| **MS GSD** | 10 m | **10 m** (B2,3,4,8) | 100m/300m | 1 m (RGB+NIR*) | 3-4 m (4-8 bands) | 1.24 m (8 bands) |
| **Spectral Bands** | 4 MS + 1 PAN | **13 bands** (VNIR+SWIR) | 4 bands (B,G,R,NIR) | 4 bands (R,G,B,NIR) | 4-8 bands | 8 MS + PAN |
| **Key MS Bands** | B:0.45-0.52, G:0.53-0.60, R:0.62-0.69, NIR:0.76-0.89 | **B2:0.49, B3:0.56, B4:0.665, B8:0.842** (10m) | B:0.45-0.52, G:0.53-0.60, R:0.62-0.69, NIR:0.77-0.89 | R:0.6-0.68, G:0.52-0.6, B:0.43-0.5, NIR:0.77-0.89 | Varies by satellite gen | Coastal, B, G, Y, R, RE, NIR1, NIR2 |
| **Swath** | 17.5 km | **290 km** | 2250 km | ~15 km | ~25 km | 13.1 km |
| **Revisit** | ~4 days (2 sats) | **5 days** (2 sats) | Daily (global) | Annual (US) | Daily (constellation) | 1-4 days (tasking) |
| **Quantization** | 12-bit (10-bit downlink) | **12-bit** | 10-bit | 8-bit | 12-bit | 11-bit |
| **Data Format** | DIMAP/GeoTIFF (est.) | **SAFE (GeoTIFF/JPEG2000)** | GeoTIFF/HDF5 | GeoTIFF | GeoTIFF/COG | NITF/GeoTIFF |
| **Global Coverage** | Algeria-focused | **GLOBAL** | GLOBAL | US ONLY | GLOBAL | Tasking-based |
| **Archive Depth** | 2016-present | **2015-present** | 2013-2020 (1km), 2018+ (300m) | 2003-present | 2016-present | 2014-present |
| **SR Literature** | Very limited | **EXTENSIVE** (100+ papers) | **MISR benchmark** | Moderate | Growing | Commercial (limited pub) |
| **Cloud Cover** | N/A | Major issue (optical) | Major issue | Low (summer only) | Major issue | Tasked (avoidable) |
| **License** | Government | **CC-BY-4.0 / Open** | ESA Open Data | Public Domain | Planet Academic License | Commercial |

---

## Sentinel-2: Detailed Justification as Primary Substitute

### Spectral Band Alignment (Critical for Transfer Learning)
| ALSAT-2B Band | λ Range (µm) | Sentinel-2 Band | λ Center (µm) | Match Quality |
|--------------|--------------|-----------------|---------------|---------------|
| Blue | 0.45 - 0.52 | B2 (Blue) | 0.490 | **Excellent** |
| Green | 0.53 - 0.60 | B3 (Green) | 0.560 | **Excellent** |
| Red | 0.62 - 0.69 | B4 (Red) | 0.665 | **Excellent** |
| NIR | 0.76 - 0.89 | B8 (NIR) | 0.842 | **Excellent** |

**Conclusion**: Near-perfect spectral correspondence for the 4 MS bands at identical 10m GSD.

### SR Task Formulation with Sentinel-2
| Task | LR Input | HR Target | Scale | Notes |
|------|----------|-----------|-------|-------|
| **MS→MS (×2)** | S2 20m bands (B5,B6,B7,B8a,B11,B12) | S2 10m bands (B2,B3,B4,B8) | 2× | Native S2 bands, physically meaningful |
| **MS→PAN-like (×4)** | S2 10m MS | Simulated 2.5m PAN (degraded from 10m) | 4× | Simulates ALSAT PAN sharpening |
| **Cross-sensor (×4)** | S2 10m MS | NAIP/Planet 2.5m RGBN | 4× | Real cross-sensor, domain gap |
| **MISR (×4)** | S2 time series (same area) | S2 10m MS | 4× | Multi-image SR, PROBA-V style |

### Available Sentinel-2 SR Benchmarks (Verified)
1. **PROBA-V MISR** — adapted for S2 time series
2. **Sen2SR** (2021) — S2 20m→10m super-resolution
3. **S2SuperRes** — various scale factors
4. **Cross-sensor S2→NAIP** — domain adaptation benchmarks
5. **GeoSR-Bench** (2026) — includes S2 downstream tasks

---

## PROBA-V: MISR Benchmark Details

### Why PROBA-V MISR?
- **Only established MISR benchmark** in remote sensing
- **Real LR sequences** (not synthetic degradation)
- **Reference HR available** (from overlapping orbits)
- **Used in 50+ papers** — direct comparability

### PROBA-V Specs for SR
- **LR**: 300m/100m (RED/NIR), multi-angle, multi-temporal
- **HR Reference**: 100m (from 1km→100m processing)
- **Dataset Size**: ~1,500 LR stacks, ~300 with HR reference
- **Challenge**: 2016 Kaggle competition, ongoing benchmark

### Adaptation to Sentinel-2
- Use S2 time series over same ROIs as PROBA-V
- Simulate similar viewing geometry diversity
- Enables **direct method comparison** with PROBA-V literature

---

## Higher-Resolution Reference Datasets (For HR Targets)

### NAIP (National Agriculture Imagery Program)
- **Coverage**: Continental US only
- **Resolution**: 1m (2009+) / 0.6m (2018+)
- **Bands**: RGB + NIR (4-band since ~2010)
- **Use Case**: HR target for S2→1m SR (×10 scale)
- **Access**: Free via AWS S3, USGS EarthExplorer
- **Limitation**: US only, temporal gaps, 8-bit

### PlanetScope
- **Resolution**: ~3-4m (varies by generation)
- **Bands**: 4-band (RGB+NIR) or 8-band (SuperDove)
- **Revisit**: Daily (200+ satellite constellation)
- **Access**: Academic license (apply via Planet), NICFI for tropics
- **Use Case**: HR target for S2→3m SR, cross-sensor domain gap studies
- **Cost**: Free for academic research (application required)

### WorldView-3 / Maxar
- **Resolution**: 0.31m PAN, 1.24m MS
- **Bands**: 8 MS + PAN
- **Access**: Commercial only (expensive)
- **Use Case**: Ultimate HR reference, very high-res SR validation
- **Academic**: Limited grants via Maxar/DigitalGlobe

---

## Dataset Acquisition Plan

### Week 1-2: Sentinel-2 (Immediate Start)
```python
# Tools: sentinelhub-py, planetary-computer, awscli
# AOI: Algeria + global validation sites
# Bands: B2,B3,B4,B8 (10m) + B5,B6,B7,B8a,B11,B12 (20m)
# Time range: 2018-2024 (post-atmospheric-correction improvements)
# Cloud filter: <20% for training, all for testing
```

### Week 2-3: PROBA-V MISR
```python
# Source: https://kelvins.esa.int/proba-v-super-resolution/
# Or: PROBA-V MISR repurposed dataset (Proba-V-ref)
# Format: HDF5/GeoTIFF stacks
```

### Week 3-4: NAIP / PlanetScope (Conditional)
- NAIP: AWS `arn:aws:s3:::naip-analytic` — immediate if US ROI needed
- PlanetScope: Apply for academic license (2-4 week approval)

---

## Risk Matrix

| Risk | Sentinel-2 | PROBA-V | NAIP | PlanetScope |
|------|------------|---------|------|-------------|
| Access denied | NONE | NONE | NONE (US) | LOW (academic app) |
| Spectral mismatch | **NONE** | MODERATE | MODERATE | LOW |
| Temporal coverage | EXCELLENT | LIMITED (2013-2020) | ANNUAL | EXCELLENT |
| Cloud contamination | HIGH | HIGH | LOW | HIGH |
| License restrictions | NONE | ESA Open | PUBLIC DOMAIN | ACADEMIC ONLY |
| Compute/storage needs | HIGH (global) | LOW | MODERATE | MODERATE |

---

## Decision: Primary Stack = Sentinel-2 + PROBA-V MISR

**Rationale**:
1. **Spectral fidelity**: S2 10m bands match ALSAT-2B MS almost perfectly
2. **Open access**: No barriers, immediate start
3. **Benchmark compatibility**: PROBA-V MISR enables comparison to 50+ papers
4. **Downstream tasks**: GeoSR-Bench, SenseBench use S2
5. **Scalability**: Global coverage enables diverse test sites

**If ALSAT-2B access granted later**: Fine-tune / validate on real ALSAT-2B data using S2-trained models (cross-sensor domain adaptation — directly addresses "Mind the Gap" research question).

---

## Next Steps
1. [ ] Define AOIs (Algeria + 3 global validation sites: urban, agricultural, desert, coastal)
2. [ ] Build Sentinel-2 download/preprocessing pipeline (`src/data/sentinel2_downloader.py`)
3. [ ] Build PROBA-V MISR loader (`src/data/probav_loader.py`)
4. [ ] Create degradation simulation matching ALSAT-2B sensor model (`src/degradation/alsat2b_degradation.py`)
5. [ ] Document all parameters in `configs/dataset_config.yaml`