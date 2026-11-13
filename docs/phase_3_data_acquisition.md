# Phase 3 Data Acquisition Report

**Date:** 2026-11-13
**Status:** Pilot acquired — full-scale acquisition pending supervisor approval
**Commit:** `5b34c08`

---

## 1. PROBA-V Configuration Corrections (Completed)

### Facts Verified Against Official Sources

| Item | Prior (Incorrect) | Corrected (Verified) | Source |
|------|-------------------|----------------------|--------|
| Dataset name | "PROBA-V Super-human Verification Challenge dataset" (fabricated) | **PROBA-V Super-Resolution dataset** | Zenodo API, ESA Kelvins |
| Region count | Audit claimed 78; config had 74 | **74** — confirmed by both sources | Zenodo `metadata.description`, ESA `/data/` page |
| Zenodo filename | `probav-v-super-resolution.zip` (inferred) | **`probav_data.zip`** | Zenodo API `files[].key` |
| File size | 675 MB (rounded from ESA page) | **708,416,603 bytes (~675 MB)** — ESA rounded, correct | Zenodo API `files[].size` |
| Checksum | `sha256:verify_after_download` (placeholder) | **md5:`7101101c7c91eeb5979d472861c1b784`** | Zenodo API |
| License | Not documented | **CC BY 4.0** | Zenodo API `metadata.license.id` |
| Download URL | `https://zenodo.org/record/.../probav-v-super-resolution.zip` | **`https://zenodo.org/api/records/6327426/files/probav_data.zip/content`** | Zenodo API direct file endpoint |

### Files Modified
- `configs/data/probav.yaml` — corrected download URL, added md5 checksum and exact size_bytes
- `docs/phase_2_scientific_audit.md` — corrected name reference
- `docs/phase_3_data_audit.md` — corrected region count finding (74 is correct; prior "78" was misreading)
- `phase_2_complete.md` — corrected completion table entry

### New Documentation
- `docs/license.md` — CC BY 4.0 attribution requirements for PROBA-V and Sentinel-2

---

## 2. Sentinel-2 Pilot Data Acquisition (Completed)

### Access Method
- **Provider:** Microsoft Planetary Computer STAC API
- **Endpoint:** `https://planetarycomputer.microsoft.com/api/stac/v1`
- **Authentication:** None required (public access)
- **Collection:** `sentinel-2-l2a` (Level-2A surface reflectance)

### AOI: DZ_01 — Algiers Urban Coastal
- **BBox:** [2.8, 36.5, 3.8, 37.0] (lon/lat WGS84)
- **Land cover:** urban_coastal
- **Period:** 2018-01-01 to 2022-12-31 (train split)
- **Cloud filter:** < 30% (lenient for pilot validation)

### Scenes Acquired

| Scene ID | MGRS Tile | Date | Cloud % | Size | Status |
|----------|-----------|------|---------|------|--------|
| S2A_MSIL2A_20221230T103431_R108_T31SEB_* | 31SEB | 2022-12-30 | 16.1% | ~665 MB | ✅ Valid |
| S2A_MSIL2A_20221230T103431_R108_T31SEA_* | 31SEA | 2022-12-30 | 6.5% | ~894 MB | ✅ Valid |
| S2A_MSIL2A_20221230T103431_R108_T31SDB_* | 31SDB | 2022-12-30 | 0.8% | ~445 MB | ✅ Valid |

**Total:** 3 scenes × 4 bands = **12 GeoTIFF files, 2,013.7 MB raw**

### Bands Downloaded
B2 (Blue, 490 nm), B3 (Green, 560 nm), B4 (Red, 665 nm), B8 (NIR, 842 nm) — all at 10 m resolution.

### Data Quality Observations
- **NaN pixels:** Present in all scenes (area outside actual scene footprint within MGRS tile). Typical coverage: 69–99% of pixel area has valid reflectance.
- **Reflectance range:** [~0.00, 1.00] after clipping — consistent with Surface Reflectance product.
- **CRS:** EPSG:4326 (WGS84) — as requested; native S2 product is UTM but reprojected for uniform coordinate space.
- **No duplicate scenes:** All three scenes have distinct MGRS tiles; same acquisition date (same overpass, different swaths).
- **Temporal leakage:** All scenes are from 2022-12-30 — this is a *pilot* validation sample, not a complete temporal split. Full acquisition will span 2018–2022.

### Storage Layout
```
data/raw/sentinel2/DZ_01/
  31SEB_20221230/
    31SEB_20221230_B2.tif   (175 MB)
    31SEB_20221230_B3.tif   (167 MB)
    31SEB_20221230_B4.tif   (162 MB)
    31SEB_20221230_B8.tif   (160 MB)
  31SEA_20221230/
    ... (similar structure)
  31SDB_20221230/
    ... (similar structure)
data/metadata/sentinel2/
  pilot_manifest.json       ← reproducible provenance record
```

### Reproducible Manifest
Generated at `data/metadata/sentinel2/pilot_manifest.json` containing:
- Source provider, endpoint, collection, license, citation
- AOI definition (id, name, bbox, land_cover)
- Temporal scope (start, end, split label)
- Per-scene: scene_id, datetime, cloud_cover, mgrs_tile, per-band file paths, sizes, MD5 checksums, shapes, bounds, geotransform
- Validation flag (`checksums_valid`: true for all files)

### Scripts Added
- `scripts/sentinel2_pilot.py` — full acquisition script (reproducible, configurable)
- `scripts/validate_pilot.py` — post-download validation against manifest
- `scripts/diag_pilot.py` — quick diagnostic for inspecting downloaded files

---

## 3. Test Results

| Test Suite | Result | Notes |
|------------|--------|-------|
| Unit tests (pytest) | **38/38 passing** | Same as before; zero regressions |
| Pilot data validation | **All 12 files valid** | Non-NaN reflectance in [0,1], CRS present, checksums match |
| Git operations | **Clean push** | Commit `5b34c08` pushed to `origin/master` |

---

## 4. Remaining Items for Full Phase 3

### Pending (Require Supervisor Approval)
- [ ] **Full Sentinel-2 acquisition**: Expand from 3 scenes to all 8 AOIs across train/val/test periods (estimated ~50+ GB)
- [ ] **Tile generation & splitting**: Apply 1024×1024 tiling with 128px overlap; enforce spatial buffer for no-leakage splits
- [ ] **Normalization statistics**: Compute per-band percentiles on train split only
- [ ] **PROBA-V full dataset access**: Download `probav_data.zip` (~675 MB) from Zenodo after small test passes

### Technical Notes
- **Planetary Computer rate limits:** No authentication needed, but heavy use may trigger throttling. Consider Copernicus Data Space Ecosystem (requires registration) for production-scale downloads.
- **Scene footprint NaNs:** ODC-stac loads the full MGRS tile extent (~10 km × 10 km at 10m), but actual S2 acquisition covers only a subset. NaN masking must be applied downstream.
- **EPSG:4326 vs UTM:** Pilot uses lon/lat (4326) for simplicity. Production tiling should use local UTM zone for accurate distance calculations.

---

## 5. Evidence of External Source Verification

1. **Zenodo API** (`GET https://zenodo.org/api/records/6327426`):
   - `metadata.title` = "PROBA-V Super-Resolution dataset"
   - `metadata.description` contains "74 hand-selected regions"
   - `metadata.license.id` = "cc-by-4.0"
   - `files[0].key` = "probav_data.zip", `size` = 708416603, `checksum` = "md5:7101101c7c91eeb5979d472861c1b784"

2. **ESA Kelvins Data Page** (`https://kelvins.esa.int/proba-v-super-resolution/data/`):
   - "**74** hand-selected regions around the globe"
   - Download link points to Zenodo record 6327426
   - File size stated as "675MB" (rounded from 675.3 MiB)

3. **Sentinel-2 Pilot**: Downloaded via Planetary Computer STAC API; source, license, and citation recorded in manifest.

---

*Report generated: 2026-11-13*
*Author: MASTER2-A1 Research Team*
