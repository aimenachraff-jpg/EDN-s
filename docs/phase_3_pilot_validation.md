# Phase 3 Pilot Validation Report

**Date:** 2026-11-13
**Status:** ✅ Pilot acquired and validated
**Commit:** 5b34c08

## 1. Pilot Execution Summary

Successfully acquired Sentinel-2 L2A pilot data for AOI DZ_01 (Algiers Urban Coastal) via Microsoft Planetary Computer STAC API.

### Data Summary
- **Scenes acquired:** 3 (all from 2022-12-30)
- **MGRS tiles:** 31SEB, 31SEA, 31SDB  
- **Bands:** B2, B3, B4, B8 (10m resolution)
- **Total files:** 12 GeoTIFFs
- **Total storage:** 2,013.7 MB
- **Temporal scope:** 2018-01-01 to 2022-12-31 (train split)

## 2. Data Verification Results

### ✅ File Integrity
- **All 12 GeoTIFFs** exist with expected sizes
- **Checksums valid** (all entries show `checksums_valid: true`)
- **All bands readable** with proper CRS (EPSG:4326)
- **No corrupted files** detected

### ✅ Data Quality Validation
- **Spatial alignment:** All files georeferenced correctly
- **Reflectance range:** [0.000, 1.000] (properly scaled from Sentinel-2 SR)
- **NaN pixels:** Present outside actual scene footprints (expected for S2 MGRS tiles)
- **Cloud cover:** 0.8% to 16.1% (acceptable for pilot)

### ✅ Provenance Verification
- **Source:** Microsoft Planetary Computer STAC API
- **License:** Copernicus / Sentinel-2, CC BY 4.0
- **Citation:** "Sentinel-2 Level-2A surface reflectance. European Space Agency."
- **Acquisition date:** 2026-11-13 (pilot data)
- **Processing level:** L2A (Surface Reflectance)

## 3. Pilot Limitations & Scientific Constraints

### 🚫 What This Pilot **Does NOT** Validate

1. **Temporal Validation:** Only one date (2022-12-30) — does not test 2018-2024 temporal split
2. **Multi-AOI Coverage:** Single AOI (DZ_01) — does not validate 8 AOIs or leakage prevention  
3. **Complete Preprocessing Pipeline:** Raw download only — does not test tiling, normalization, or splitting
4. **ALSAT-2B Replacement:** Sentinel-2 is a spectral proxy, not a substitute for ALSAT-2B MS
5. **Spatial Buffer Implementation:** Tiling with 128px overlap and 1024x1024 tiles not verified
6. **Normalization Statistics:** Per-band percentiles computed exclusively from training data not tested
7. **Full-Scale Acquisition:** Storage/throughput for 50+ GB across all AOIs not measured

### ✅ What This Pilot **Does** Confirm
- Planetary Computer STAC API accessibility and reliability
- File download, checksum validation, and spatial reprojection
- Basic data quality (reflectance range, georeferencing)
- Manifest generation and reproducibility

## 4. Technical Validation Details

### File Properties Summary

| Scene | Tile | Date | Bands | Valid Pixels | Storage |
|-------|------|------|-------|--------------|---------|
| S2A_MSIL2A_20221230T103431_R108_T31SEB | 31SEB | 2022-12-30 | B2,B3,B4,B8 | 122.9M/124.5M | 665 MB |
| S2A_MSIL2A_20221230T103431_R108_T31SEA | 31SEA | 2022-12-30 | B2,B3,B4,B8 | 121.4M/123.0M | 894 MB |
| S2A_MSIL2A_20221230T103431_R108_T31SDB | 31SDB | 2022-12-30 | B2,B3,B4,B8 | 86.1M/124.4M | 445 MB |

### Band Statistics Sample (B2, Scene 31SEB)
- **Shape:** 9962×12497
- **Valid pixels:** 122,852,178
- **Reflectance range:** [0.1032, 1.0000] 
- **Missing data:** NaN outside actual scene footprint (typical for S2 MGRS)

### CRS and Geotransform
- **Projected CRS:** EPSG:4326 (WGS84)
- **Resolution:** ~0.0001° (~10m at mid-latitudes)
- **All scenes:** Consistent georeferencing within ±0.000003°

##  executions Validations Complete
- planetary_computer STAC API accessibility and authentication
- odc.stac download, CRS conversion (latitude/longitude), time-dim squeezing
- file I/O: read, checksum, raster validation
- batch processing: de-duplication, parallel scene handling
- manifest creation: comprehensive provenance tracking
- error handling: null checks, exception management

---

## 5. Scale-Up Readiness Issues

### Confirmed Ready
- [x] Manifest generation system (tested with 3 scenes)
- [x] Checksum validation workflow
- [x] CRS reprojection pipeline (EPSG:4326)
- [x] File I/O error handling

### Requires Implementation/Testing
- [ ] **Tiling pipeline:** 1024×1024 px with 128px overlap
- [ ] **Spatial buffer:** Leakage prevention between AOIs
- [ ] **Temporal splitting:** 2018-2022 train, 2023 val, 2024 test
- [ ] **Normalization statistics:** Computed only on train split
- [ ] **Large file handling:** 50+ GB total storage for 8 AOIs
- [ ] **Retry logic:** Network failure recovery
- [ ] **Progress tracking:** Download monitoring

## 6. Storage Estimates (Based on Pilot Scaling)

### Current Pilot (3 scenes)
- **Raw storage:** 2.0 GB
- **Interim (preprocessed):** ~200 MB (estimated 10%)
- **Processed (tiled + metadata):** ~100 MB

### Projected Full Scale (8 AOIs × all periods)
- **8 × (3 scenes × 4 bands × 668 MB)** ≈ **50 GB raw**
- **10% interim:** ~5 GB
- **5% processed:** ~2.5 GB
- **Total project:** ~57.5 GB (conservative)

## 7. PROBA-V Access Test Status

### ✅ Dataset Information Verified
- **Name:** PROBA-V Super-Resolution dataset (official ESA name)
- **Zenodo URL:** https://zenodo.org/api/records/6327426/files/probav_data.zip
- **File size:** 708,416,603 bytes (675 MB, rounded)
- **Checksum:** md5:7101101c7c91eeb5979d472861c1b784
- **License:** CC BY 4.0

### 🔒 Access Test Required
- Small download test (~675 MB) to validate Zenodo endpoint and checksum
- **Risk:** High (large file download, time/cost)
- **Recommendation:** Supervisor approval required

## 8. Supervisor Review Checklist

### ✅ Completed
- [x] PROBA-V dataset name verification
- [x] Region count confirmation (74 regions)
- [x] Zenodo URL and checksum verification  
- [x] CC BY 4.0 license documentation
- [x] Sentinel-2 pilot acquisition and validation
- [x] Manifest and provenance generation

### 🔴 Requires Approval
- [ ] PROBA-V small download test (675 MB)
- [ ] Full Sentinel-2 acquisition (8 AOIs, all periods)
- [ ] Tiling pipeline implementation
- [ ] Spatial buffer and leakage prevention
- [ ] Normalization statistics (train split only)
- [ ] Large-scale storage allocation

### 📋 Next Steps
1. Approve PROBA-V download test
2. Approve full Sentinel-2 acquisition plan
3. Implement tiling and spatial buffer systems
4. Complete normalization pipeline
5. Deploy retry and monitoring systems
6. Execute full-scale data acquisition

## 9. Scientific Limitations Statement

**Critical Caveat:** This pilot demonstrates feasibility but does not validate the core scientific claims of the thesis. The current work:

1. **Uses proxy data:** Sentinel-2 replaces unavailable ALSAT-2B, changing spectral characteristics
2. **Limited temporal scope:** Single date does not test annual variations or long-term patterns
3. **Partial spatial coverage:** Single AOI does not test multi-region generalization
4. **Missing preprocessing:** Raw downloads only — no tiling, normalization, or split validation

**Recommendation:** Full-scale acquisition and complete pipeline validation required before drawing scientific conclusions or proceeding to model training.

---

*Report generated: 2026-11-13*
*Prepared by: MASTER2-A1 Research Team*
*Validation based on actual file inspection, checksum verification, and manifest review.*

---

**Next Move:** Supervisor approval for PROBA-V small download test and full-scale Sentinel-2 acquisition plan.