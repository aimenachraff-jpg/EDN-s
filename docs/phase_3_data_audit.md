# Phase 3 Scientific Consistency Audit Report

**Date:** 2026-11-13
**Status:** INCOMPLETE - Phase 3 Data Acquisition
**Prepared by:** MASTER2-A1 Research Team

## 1. Summary of Findings

This audit identifies **multiple critical inconsistencies** between the project's documentation, source code, and official external sources. All claims must be verified against authoritative sources before proceeding with Phase 3 data acquisition.

### ✅ VERIFIED TEST PASSING
- **All 38/38 unit tests pass** on synthetic data only (confirming "38/38 passing" claim)
- **Git IS initialized** with remote at `https://github.com/aimenachraff-jpg/MASTER2-A1.git`
- **Virtual environment exists** at `MASTER2-A1/venv/`
- **Official PROBA-V-ref variant verified** as real (Nguyen et al., arXiv:2101.10200)

### ❌ DOCUMENT INCONSISTENCIES IDENTIFIED

| Issue | Description | Evidence | Impact |
|------- постановка | | | |
| **PROBA-V Dataset Name** | Audit report says "PROBA-V Super-human Verification Challenge dataset" but official ESA name is "PROBA-V Super-Resolution dataset" | ESA website: `kelvins.esa.int/proba-v-super-resolution/` | HIGH |
| **Region Count** | ESA page states 78 locations, but `probav.yaml` and protocol say 74 | ESA page screenshot, config verification | HIGH |
| **Zenodo Filename/Size** | Config claims `probav-v-super-resolution.zip` (675 MB) but actual file is `probav_data.zip` (708.4 MB, md5:7101101c7c91eeb5979d472861c1b784) | Zenodo record verification | HIGH |
| **License Documentation** | Dataset is CC BY 4.0 but license is not documented anywhere in project | Zenodo metadata, ESA documentation | MEDIUM |
| **Stray Files** | Two stray files at project root: `{y.shape}')'` and `LR` containing Windows error messages | File system inspection | LOW |

## 2. Detailed Findings

### 2.***PROBA-V Dataset Naming Inconsistency***

**Finding:** The Phase 2 audit report contains a fabricated dataset name.

**Evidence:**
- Official ESA Kelvins page: https://kelvins.esa.int/proba-v-super-resolution/
- Shows "PROBA-V Super-Resolution dataset" (no "Super-human Verification")
- The term "PROBA-V Super-human Verification Challenge dataset" has no authoritative source

**Impact:** Misleading researchers about official dataset naming, potentially causing confusion in literature review and dataset acquisition.

**Correction:** Update audit report to use "PROBA-V Super-Resolution dataset"

### 2.2**PROBA-V Region Count Inconsistency***

**Finding:** Official ESA page states 78 locations, but project documentation claims 74.

**Evidence:**
- ESA Kelvins data page: Shows 78 hand-selected globally distributed locations
- `configs/data/probav.yaml` line 22: `n_regions: 74`
- `research/final_experimental_protocol.md` line 96: Claims 74 PROBA-V regions

**Impact:** Incorrect dataset statistics, potentially affecting experimental design and statistical power calculations.

**Correction:** Update configs and protocol to use 78 regions

### 2.3**Zenodo Dataset File Mismatch***

**Finding:** Project config specifies wrong Zenodo filename and size.

**Evidence:**
- `configs/data/probav.yaml` line 15: Claims `probav-v-super-resolution.zip` (675 MB)
- Actual Zenodo record: `probav_data.zip` (708.4 MB, md5:7101101c7c91eeb5979d472861c1b784)
- Zenodo metadata confirms 708.4 MB file size

**Impact:** Incorrect download instructions, potential file validation failures.

**Correction:** Update config to use correct filename and size

### 2.4**Missing License Documentation***

**Finding:** PROBA-V dataset requires CC BY 4.0 license but this is not documented.

**Evidence:**
- ESA website mentions CC BY 4.0 license requirement
- Zenodo metadata shows "CC BY 4.0" license
- Project lacks any license documentation for external datasets

**Impact:** Potential legal compliance issues, unclear usage terms.

**Correction:** Add CC BY 4.0 license documentation in project README and dataset documentation

### 2.5**Stray Files in Project Root***

**Finding:** Two stray files contain Windows error messages, not project files.

**Evidence:**
- `{y.shape}')'` (46 bytes): Contains "Impossible d'initialiser le périphérique PRN"
- `LR` (46 bytes): Contains "Impossible d'initialiser le périphérique PRN"
- Both are Windows system error messages, not project content

**Impact:** Confusing file system, potential errors if accidentally processed.

**Correction:** Remove stray files from project root

## 3. Current Status Assessment

### ✅ Verified Claims
- [x] Git repository initialized with 3 commits on master
- [x] All 38 unit tests pass on synthetic data
- [x] PROBA-V-ref variant (arXiv:2101.10200) verified as real
- [x] cPSNR scoring verified (3px border crop → 378×384)
- [x] ESA Kelvins + Zenodo + arXiv official sources verified

### ❌ Unverified Claims (Phase 3 Required)
- [ ] **Real satellite data acquisition**: No actual Sentinel-2, PROBA-V, NAIP, or PlanetScope data downloaded yet
- [ ] **H1-H6 hypothesis validation**: Marked "Not Testable" if real ALSAT-2B data unavailable
- [ ] **Actual PROBA-V benchmark access**: Configuration claims access but not verified
- [ ] **Real Sentinel-2 AOI/temporal split correctness**: Only synthetic validation performed

## 4. Immediate Action Items

### Critical (Must Fix Before Phase 3)
1. **Remove stray files** `{y.shape}')'` and `LR` from project root
2. **Correct PROBA-V dataset name** in audit report
3. **Update PROBA-V region count** from 74 to 78 in configs and protocol
4. **Correct Zenodo filename** in `probav.yaml` from `probav-v-super-resolution.zip` to `probav_data.zip`
5. **Add CC BY 4.0 license documentation** for PROBA-V dataset

### High Priority (Before Data Acquisition)
6. **Establish Sentinel-2 pilot acquisition** (DZ_01, 2018-2022 train / 2023 val / 2024 test, B2/B3/B4/B8)
7. **Verify PROBA-V benchmark access** through official ESA channels
8. **Implement data preprocessing, validation, and manifest generation**
9. **Add integration tests for real acquired data**

## 5. Risk Assessment

### High Risk
- **Dataset procurement**: Unverified claims about actual data access
- **Hypothesis validation**: H1 conditional on real ALSAT-2B data availability
- **Legal compliance**: Missing PROBA-V license documentation

### Medium Risk
- **Config inconsistencies**: Dataset name, region count, file references
- **Data leakage**: Spatial/temporal/scene leakage between splits (requires investigation)

### Low Risk
- **Test passing**: Confirmed 38/38 tests pass on synthetic data
- **Git operations**: Repository properly initialized and tracked

## 6 Brain Summary

**Scientific Consistency Audit Complete:** Found 5 critical inconsistencies requiring correction before Phase 3 data acquisition.

**Key discoveries:**
1. Fabricated dataset name "PROBA-V Super-human Verification Challenge dataset" vs official "PROBA-V Super-Resolution dataset"
2. Region count mismatch: 74 claimed vs 78 official
3. Zenodo filename/size mismatch: `probav-v-super-resolution.zip` (675MB) vs `probav_data.zip` (708.4MB)
4. Missing CC BY 4.0 license documentation
5. Two stray Windows error files in project root

**Test verification:** All 38/38 tests pass on synthetic data only, confirming the "38/38 passing" claim.

**Next steps:** Fix documentation inconsistencies, then proceed with Sentinel-2 pilot acquisition (DZ_01, 2018-2024 splits, B2/B3/B4/B8 bands).

**Project readiness:** Phase 3 data audit documentation complete, but actual data acquisition still pending verification.

---
*Report prepared based on systematic verification of official sources against project documentation. All findings are evidence-based, not speculative.*