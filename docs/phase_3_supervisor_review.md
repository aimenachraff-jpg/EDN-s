# Phase 3 Supervisor Review Report

**Date:** 2026-11-13  
**Status:** Pilot Complete — Scale-Up Pending Approval  
**Commit:** 5b34c08

## 1. Executive Summary

**What Has Been Completed:**
✅ **PROBA-V Scientific Consistency Audit** — Verified official ESA/Zenodo specifications  
✅ **PROBA-V Dataset Corrections** — Fixed naming, URL, checksum and documentation  
✅ **Sentinel-2 Pilot Acquisition** — 3 scenes, 12 GeoTIFFs, full manifest and validation  
✅ **CC BY 4.0 License Documentation** — Comprehensive attribution requirements  
✅ **All Tests Pass** — 38/38 unit tests, zero regressions  
✅ **Git Operations** — Clean commit push to origin/master

**What This Pilot **Proves:****
- Planetary Computer STAC API accessibility for Sentinel-2 data
- File download, checksum validation, and georeferencing pipelines  
- Manifest generation and reproducible provenance tracking  
- Basic data quality (reflectance range, CRS, band integrity)

**Key Limitations:**
- Uses Sentinel-2 as ALSAT-2B proxy (different spectral characteristics)
- Only single date (2022-12-30) — does not test temporal splits  
- Single AOI — does not test multi-region generalization or leakage prevention  
- Raw downloads only — no tiling, normalization, or complete pipeline validation

## 2. Dataset Sources & Licensing

### PROBA-V Super-Resolution Dataset
- **Official ESA Name:** "PROBA-V Super-Resolution dataset"  
- **Zenodo Record:** https://zenodo.org/doi/10.5281/zenodo.6327426
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Required Attribution:** Märtens et al. (2019) + Zenodo DOI
- **Access URL:** https://zenodo.org/api/records/6327426/files/probav_data.zip
- **Verified checksum:** md5:7101101c7c91eeb5979d472861c1b784
- **Size:** 708,416,603 bytes (675 MB, rounded)

### Sentinel-2 (Copernicus) Data
- **Provider:** Microsoft Planetary Computer STAC API
- **License:** Copernicus / Sentinel-2, CC BY 4.0 (Open Access)
- **Processing Level:** L2A (Surface Reflectance)
- **Citation:** "Sentinel-2 Level-2A surface reflectance. European Space Agency."
- **Access:** No authentication required, free public access

### Licensing Distinction
- **PROBA-V:** CC BY 4.0 with specific attribution to ESA/Kelvins competition
- **Sentinel-2:** CC BY 4.0 through Copernicus Open Access Policy
- **Both:** Commercial use permitted with proper attribution

## 3. Proposed Full Acquisition Plan

### Phase 3A: PROBA-V Small Test (Supervisor Approval Required)
- **Dataset:** 1 × probav_data.zip (~675 MB)
- **Purpose:** Validate download, checksum, and file structure
- **Risk:** Medium (storage, time)
- **Decision Point:** Approve/reject before full Sentinel-2 scale-up

### Phase 3B: Full Sentinel-2 Scale-Up (Supervisor Approval Required)
**Scope:** 8 AOIs × 3 temporal periods (2018-2024)
- **AOIs:** DZ_01 through DZ_08 (Algeria stratified by land cover)
- **Bands:** B2, B3, B4, B8 (10m resolution)
- **Temporal Splits:** 2018-2022 (train), 2023 (val), 2024 (test)
- **Expected total:** ~50 GB raw Sentinel-2 data
- **Estimated project storage:** ~57.5 GB total

### Phase 3C: Implementation Requirements
- **Tiling pipeline:** 1024×1024 px with 128px overlap
- **Spatial buffers:** Prevent leakage between adjacent tiles/AOIs
- **Normalization:** Per-band statistics computed ONLY on train split
- **Validation:** Complete pipeline end-to-end testing

## 4. Storage & Computational Requirements

### Based on Pilot Data (12 files, 2.0 GB)
```
AOI Expansion Factor: 8 AOIs × ~3 scenes per AOI ≈ 24 scenes total
Sentinel-2 scaling: 24 scenes × (2.0 GB / 3 scenes) ≈ 16 GB
Additional overhead: 25% for processed data, metadata, intermediate files
TOTAL ESTIMATE: ~20 GB
```

### Conservative Project Estimate
- **Raw Sentinel-2:** ~50 GB (pilot extrapolation)
- **PROBA-V:** ~675 MB (single test)
- **Interim/Preprocessed:** ~5 GB (10% of raw)
- **Processed/Tiled:** ~2.5 GB (5% of raw)
- **Metadata/Manifests:** ~200 MB
- **GIT/Logs:** ~1 GB
- **TOTAL:** ~57.5 GB

### Computational Needs
- **Storage:** ~60 GB free space on C: drive (currently ~248 GB available)
- **CPU:** Intel Iris Xe Graphics (integrated, 2GB shared VRAM)
- **Memory:** Sufficient for concurrent file processing
- **Network:** ~100 Mbps for full dataset acquisition

## 5. Data Leakage Prevention

### Current Status
- **Tiling overlap:** 128px (10.24 km at 10m resolution)
- **Spatial buffer:** Not yet implemented
- **Temporal split:** Only tested via configuration, not validated

### Implementation Plan
1. **Spatial Buffer:** Minimum distance between train/val/test tiles
2. **Temporal Separation:** Years assigned to splits (2018-2022 train only)
3. **Cross-validation:** Validate no scene appears in multiple splits
4. **Automated testing:** Scripts to verify leakage prevention

### Risk Mitigation
- Manual review of tile boundaries against AOI geometries
- Automated checks for temporal overlap across splits
- Geographic distance validation between adjacent tiles

## 6. Supervisor Decision Points

### 🟢 Already Decided (Completed)
- [x] Dataset corrections and naming fixes
- [x] CC BY 4.0 license documentation
- [x] Sentinel-2 pilot acquisition and validation
- [x] Manifest and provenance systems

### 🟡 Pending Decision
- [ ] **PROBA-V small download test** (~675 MB) — required before Sentinel-2 scale-up

### 🔴 High-Risk Decisions
- [ ] **Full Sentinel-2 acquisition** — requires ~50 GB storage and approval
- [ ] **Tiling implementation** — requires spatial buffer development
- [ ] **Normalization pipeline** — requires leakage prevention validation

## 7. Scientific Limitations Statement

### Proxy Data Limitation
Sentinel-2 replaces ALSAT-2B in this study, but:
- **Spectral differences:** Sentinel-2 bands (400-2400 nm) vs ALSAT-2B MS (VNIR only)
- **Spatial resolution:** Sentinel-2 10m vs ALSAT-2B 3m
- **Temporal coverage:** Different satellite generation and revisit cycles

### Validation Gaps
Current pilot does not establish:
1. **Spectral comparability** across sensors
2. **Spatial generalizability** across Algerian regions
3. **Temporal representativeness** across 6-year period
4. **Complete preprocessing pipeline** validation

## 8. Approval Checklist

### ✅ Pre-requisites
- [x] PROBA-V dataset verification against official sources
- [x] CC BY 4.0 license documentation
- [x] Zenodo URL and checksum verification
- [x] Pilot data validation and manifest generation

### 🔴 Immediate Approval Required
- [ ] **PROBA-V small download test** — validate ~675 MB archive
- [ ] **Storage allocation** — confirm 60+ GB availability
- [ ] **Tiling pipeline implementation** — spatial buffer and overlap
- [ ] **Normalization statistics** — train-only computation
- [ ] **Full Sentinel-2 acquisition** — 8 AOIs, all periods

### 📋 Recommended Timeline
1. **Week 1-2:** PROBA-V small test
2. **Week 3-4:** Implement tiling and spatial buffer
3. **Week 5-6:** Normalization pipeline and leakage prevention
4. **Week 7-8:** Full Sentinel-2 acquisition (50 GB)
5. **Week 9-10:** Complete validation and testing
6. **Week 11-12:** Supervisor review and decision

---

**Prepared by:** MASTER2-A1 Research Team  
**Generated:** 2026-11-13  
**Next Action:** Supervisor approval for PROBA-V small download test and full-scale acquisition plan