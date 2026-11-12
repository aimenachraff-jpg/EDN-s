# Dataset Investigation: ALSAT-2B and Substitute Datasets

## 1. ALSAT-2B / NAOMI Instrument — Verified Specifications

**Source**: ESA eoPortal (https://www.eoportal.org/satellite-missions/alsat-2), last updated Nov 7, 2025

### Mission Parameters
| Parameter | Value |
|-----------|-------|
| **Satellites** | AlSat-2A (launch 12 Jul 2010), AlSat-2B (launch 26 Sep 2016) |
| **Operator** | Algerian Space Agency (ASAL) |
| **Orbit** | Sun-synchronous, 670 km, 98.2° inclination, 10:30 LTAN |
| **Mission Life** | 5 years design, extended operations |
| **Status** | Operational (extended) |

### NAOMI Instrument — Optical Design
| Parameter | Value |
|-----------|-------|
| **Type** | Pushbroom imager |
| **Telescope** | Korsch Three-Mirror Anastigmat (TMA), Silicon Carbide (SiC) |
| **Aperture** | 200 mm |
| **Focal Length** | 3.2 m (F/16) |
| **FOV** | 1.5° (cross-track) |
| **Swath** | 17.5 km at nadir |

### Detector Arrays
| Parameter | Panchromatic (PAN) | Multispectral (MS) × 4 bands |
|-----------|-------------------|------------------------------|
| **Detector** | CCD, TDI (Time Delay Integration) | CCD, 4 separate arrays |
| **Pixels (cross-track)** | 7,000 | 1,750 per band |
| **TDI Stages** | Up to 64 (adaptive) | Up to 32 (adaptive) |
| **Pixel Pitch** | 6.5 µm | 6.5 µm |
| **Dynamic Range** | 12-bit (4096 levels) | 12-bit (4096 levels) |
| **Downlink Quantization** | 10-bit (1024 levels) | 10-bit (1024 levels) |

### Spectral Bands (Verified)
| Band | Name | Spectral Range (µm) | GSD | Primary Use |
|------|------|---------------------|-----|-------------|
| **PAN** | Panchromatic | 0.45 – 0.75 | **2.5 m** | High-res grayscale, cartography |
| **B1** | Blue | 0.45 – 0.52 | **10 m** | Coastal/aerosol, vegetation |
| **B2** | Green | 0.53 – 0.60 | **10 m** | Vegetation, water |
| **B3** | Red | 0.62 – 0.69 | **10 m** | Vegetation, soil |
| **B4** | NIR | 0.76 – 0.89 | **10 m** | Vegetation (NDVI), water |

### Radiometric Performance
| Parameter | PAN | MS |
|-----------|-----|-----|
| **SNR (at reference radiance)** | > 100 | > 100 |
| **MTF (at Nyquist)** | > 0.15 | > 0.15 |
| **Radiometric Accuracy** | < 5% | < 5% |
| **Inter-band Registration** | N/A | < 0.3 pixel |
| **Geo-location Accuracy** | < 10 m (with GCPs) | < 10 m (with GCPs) |

### Data Products
| Product Level | Description | Format | Typical Size |
|---------------|-------------|--------|--------------|
| **Level 1A** | Raw detector data + telemetry | DIMAP/GeoTIFF | ~500 MB/scene |
| **Level 1B** | Radiometrically corrected, sensor geometry | DIMAP/GeoTIFF | ~500 MB/scene |
| **Level 1C** | Orthorectified, map projection (UTM/WGS84) | DIMAP/GeoTIFF | ~500 MB/scene |
| **Level 2A** | Atmospherically corrected (surface reflectance) | DIMAP/GeoTIFF | ~500 MB/scene |

---

## 2. ALSAT-2B Data Access — Current Status

### Official Channels
| Channel | Status | Details |
|---------|--------|---------|
| **ASAL Website (asal.dz)** | **INACCESSIBLE** | Connection refused/timeout from multiple locations |
| **ASAL Data Portal** | **NOT FOUND** | No public catalog discovered |
| **ESA eoPortal** | METADATA ONLY | Links to ASAL for data requests |
| **CNES/PEPS** | NOT CONTACTED | Potential via French partnership (Airbus built satellite) |
| **ESA TPM (Third Party Missions)** | NOT CONTACTED | ALSAT-2 may be registered |
| **International Charter** | DISASTER ONLY | Not for research |

### Likely Access Process (Based on Typical National Mission Policies)
1. **Formal Request**: Research proposal to ASAL (French/Arabic)
2. **Institutional Agreement**: MoU between university and ASAL
3. **Data License**: Restricted use, no redistribution, acknowledgment required
4. **Delivery**: FTP/secure download or physical media
5. **Timeline**: 2-6 months typical

### Contact Information Needed
- ASAL Earth Observation department email
- CNES PEPS helpdesk (peps.cnes.fr)
- ESA TPM office (earth.esa.int/eogateway/)

---

## 3. Substitute Datasets — Detailed Technical Specs

### 3.1 Sentinel-2 (Primary Substitute) — COPERNICUS OPEN ACCESS

**Source**: ESA Sentinel-2 User Guide, Copernicus Open Access Hub

| Parameter | Value |
|-----------|-------|
| **Constellation** | Sentinel-2A (2015), Sentinel-2B (2017), Sentinel-2C (2024 planned) |
| **Orbit** | Sun-synchronous, 786 km, 98.6°, 10:30 LTAN |
| **Revisit** | 5 days (2 satellites), 2-3 days (3 satellites) |
| **Swath** | 290 km |
| **Instrument** | MSI (MultiSpectral Instrument), pushbroom |
| **Quantization** | 12-bit (4096 levels), JPEG2000 compressed |

#### Spectral Bands (10m and 20m Groups)
| Band | Name | Central λ (nm) | Bandwidth (nm) | GSD | ALSAT-2B Match |
|------|------|----------------|----------------|-----|----------------|
| **B1** | Coastal Aerosol | 443 | 21 | 60 m | — |
| **B2** | **Blue** | **490** | **66** | **10 m** | **B1 (0.45-0.52) — EXCELLENT** |
| **B3** | **Green** | **560** | **36** | **10 m** | **B2 (0.53-0.60) — EXCELLENT** |
| **B4** | **Red** | **665** | **31** | **10 m** | **B3 (0.62-0.69) — EXCELLENT** |
| **B5** | Vegetation Red Edge 1 | 705 | 15 | 20 m | — |
| **B6** | Vegetation Red Edge 2 | 740 | 15 | 20 m | — |
| **B7** | Vegetation Red Edge 3 | 783 | 20 | 20 m | — |
| **B8** | **NIR** | **842** | **115** | **10 m** | **B4 (0.76-0.89) — EXCELLENT** |
| **B8A** | Narrow NIR | 865 | 20 | 20 m | — |
| **B9** | Water Vapor | 945 | 20 | 60 m | — |
| **B10** | SWIR Cirrus | 1375 | 30 | 60 m | — |
| **B11** | **SWIR 1** | **1610** | **91** | **20 m** | — |
| **B12** | **SWIR 2** | **2190** | **175** | **20 m** | — |

#### Key SR Formulations Enabled
| Formulation | LR Input | HR Target | Scale | Physical Meaning |
|-------------|----------|-----------|-------|------------------|
| **MS-Upscaling** | B5,B6,B7,B8a,B11,B12 (20m) | B2,B3,B4,B8 (10m) | 2× | Native S2, vegetation monitoring |
| **PAN-Sharpening Sim** | B2,B3,B4,B8 (10m) | Synthetic 2.5m PAN | 4× | Matches ALSAT PAN/MS ratio |
| **Cross-Sensor** | S2 10m MS | NAIP 1m / Planet 3m | 3-10× | Domain gap, real HR |
| **MISR** | S2 Time Series (same ROI) | S2 10m MS | 2-4× | Multi-temporal, PROBA-V style |

#### Data Access
- **Copernicus Data Space Ecosystem** (dataspace.copernicus.eu) — primary
- **AWS S3** (sentinel-s2-l2a) — requester pays
- **Planetary Computer** (planetarycomputer.microsoft.com) — STAC API
- **sentinelhub-py**, **planetary-computer**, **odc-stac** — Python libraries

#### Processing Levels
- **L1C**: Top-of-atmosphere, orthorectified
- **L2A**: Bottom-of-atmosphere (surface reflectance), scene classification — **USE THIS**

---

### 3.2 PROBA-V (MISR Benchmark) — ESA OPEN DATA

**Source**: PROBA-V Mission, ESA Kelvin Platform

| Parameter | Value |
|-----------|-------|
| **Mission** | Project for On-Board Autonomy - Vegetation |
| **Launch** | 2013, End of life 2020 (1km), 300m continued |
| **Instrument** | VEGETATION imager (VNIR) |
| **Orbit** | Sun-synchronous, 820 km, 98.7°, 10:30 LTAN |
| **Swath** | 2250 km (global daily) |

#### Spectral Bands
| Band | λ Range (µm) | GSD (1km mission) | GSD (300m mission) |
|------|--------------|-------------------|---------------------|
| **BLUE** | 0.45 – 0.52 | 1 km | **300 m** |
| **RED** | 0.62 – 0.69 | 1 km | **300 m** |
| **NIR** | 0.77 – 0.89 | 1 km | **300 m** |
| **SWIR** | 1.55 – 1.65 | 1 km | 1 km (no 300m) |

#### MISR Dataset (PROBA-V Super-Resolution Challenge)
- **LR**: 300m multi-angle, multi-temporal stacks (9-15 images per stack)
- **HR Reference**: 100m from overlapping orbits (limited locations)
- **Size**: ~1,500 LR stacks, ~300 with HR
- **Challenge**: 2016 Kaggle, ongoing benchmark
- **Access**: ESA Kelvin platform, Proba-V-ref repurposed dataset

---

### 3.3 NAIP — USDA PUBLIC DOMAIN

| Parameter | Value |
|-----------|-------|
| **Coverage** | Continental US (annual state cycles) |
| **Resolution** | 1m (2009+), 0.6m (2018+) |
| **Bands** | RGB (3-band), RGB+NIR (4-band since ~2010) |
| **Spectral** | R:0.6-0.68, G:0.52-0.6, B:0.43-0.5, NIR:0.77-0.89 |
| **Format** | GeoTIFF, MrSID, JP2 |
| **Access** | AWS S3 (`arn:aws:s3:::naip-analytic`), USGS EarthExplorer |
| **License** | Public Domain |

---

### 3.4 PlanetScope — PLANET LABS (ACADEMIC LICENSE)

| Parameter | Value |
|-----------|-------|
| **Constellation** | 200+ Doves (3-4m), SuperDoves (8-band) |
| **Resolution** | ~3.7m (Dove Classic), ~3m (SuperDove) |
| **Bands** | 4-band (RGB+NIR) or 8-band (Coastal, B, G, Y, G, R, RE, NIR) |
| **Revisit** | Daily (global) |
| **Access** | Academic license application (2-4 weeks), NICFI for tropics |
| **Format** | COG (Cloud Optimized GeoTIFF), analytic/surface reflectance |
| **License** | Planet Academic License (non-commercial research) |

---

### 3.5 WorldView-3 — MAXAR (COMMERCIAL)

| Parameter | Value |
|-----------|-------|
| **Resolution** | 0.31m PAN, 1.24m MS, 3.7m SWIR, 30m CAVIS |
| **Bands** | 8 MS + PAN + 8 SWIR + CAVIS |
| **Access** | Commercial purchase, limited academic grants |
| **Use Case** | Ultimate HR reference, validation only |

---

## 4. Degradation Modeling for ALSAT-2B Simulation

### Sensor Degradation Pipeline (to apply to HR reference data)
```python
# 1. Spectral Response Function (SRF) Convolution
#    - Integrate HR hyperspectral / multi-band through ALSAT SRFs
#    - SRFs: approximate from band edges (trapezoidal) or request from ASAL

# 2. Spatial Degradation (MTF + Downsampling)
#    - MTF: modeled as Gaussian (σ from MTF@Nyquist > 0.15)
#    - Downsampling: bicubic/area averaging to target GSD

# 3. Radiometric Degradation
#    - Quantization: 12-bit → 10-bit (simulate downlink)
#    - Noise: Poisson (photon) + Gaussian (read) matching SNR>100
#    - Non-linearity: if known from calibration

# 4. Geometric Distortions
#    - Pushbroom jitter, smile/keystone (if characterized)
#    - Co-registration error: <0.3 pixel inter-band
```

### Synthetic LR Generation Strategy
| Approach | Pros | Cons |
|----------|------|------|
| **Wald Protocol** (standard) | Reproducible, matches sensor physics | Requires accurate MTF/SRF |
| **Bicubic Downsampling** | Simple, baseline | Unrealistic, no aliasing |
| **Real Degradation (PROBA-V style)** | Realistic, no simulation bias | Limited to available data |
| **Cross-Sensor (S2→ALSAT sim)** | Domain gap study | Compound errors |

**Recommendation**: Use **Wald Protocol** with estimated ALSAT-2B MTF/SRF for primary experiments; validate with **real cross-sensor** (S2→NAIP/Planet) for domain gap analysis.

---

## 5. Dataset Selection Decision Matrix

| Criterion | Weight | Sentinel-2 | PROBA-V | NAIP | PlanetScope |
|-----------|--------|------------|---------|------|-------------|
| Spectral match to ALSAT-2B MS | 0.30 | **1.0** | 0.7 | 0.8 | 0.85 |
| GSD match (10m MS) | 0.20 | **1.0** | 0.0 | 0.0 | 0.3 |
| Open access / no barriers | 0.15 | **1.0** | **1.0** | **1.0** | 0.6 |
| Global coverage | 0.10 | **1.0** | **1.0** | 0.0 | **1.0** |
| Existing SR benchmarks | 0.10 | **1.0** | **1.0** | 0.5 | 0.3 |
| Downstream task benchmarks | 0.10 | **1.0** | 0.5 | 0.3 | 0.3 |
| MISR capability | 0.05 | 0.8 | **1.0** | 0.0 | 0.2 |
| **WEIGHTED SCORE** | 1.00 | **0.97** | 0.67 | 0.51 | 0.58 |

**DECISION**: Sentinel-2 + PROBA-V MISR as primary stack.

---

## 6. Implementation Checklist

### Data Pipeline (src/data/)
- [ ] `sentinel2_downloader.py` — STAC search, download L2A, band selection
- [ ] `probav_loader.py` — MISR stack loading, HR reference alignment
- [ ] `naip_downloader.py` — AWS S3 sync for US ROIs (conditional)
- [ ] `planet_downloader.py` — Planet API client (conditional)
- [ ] `alsat2b_simulator.py` — Wald degradation with estimated MTF/SRF
- [ ] `dataset_config.yaml` — All parameters versioned

### AOI Definition (configs/aois.yaml)
- [ ] **Primary**: Algeria (3-5 diverse regions: Algiers urban, Sahara desert, Tell Atlas agriculture, coastal)
- [ ] **Validation 1**: Urban (Paris, France / Los Angeles, US)
- [ ] **Validation 2**: Agricultural (Iowa US / Punjab India)
- [ ] **Validation 3**: Desert (Sahara / Atacama)
- [ ] **Validation 4**: Coastal (Mediterranean / Gulf of Mexico)

### Temporal Strategy
- [ ] Training: 2018-2022 (S2 L2A mature)
- [ ] Validation: 2023
- [ ] Test: 2024 (held out)
- [ ] Cloud filter: <20% for train/val, all for test (with cloud mask)

---

## 7. References (Verified)

1. **ESA eoPortal ALSAT-2**: https://www.eoportal.org/satellite-missions/alsat-2 (accessed 2025-11-07)
2. **Sentinel-2 User Guide**: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
3. **PROBA-V MISR Challenge**: https://kelvins.esa.int/proba-v-super-resolution/
4. **Copernicus Data Space**: https://dataspace.copernicus.eu/
5. **NAIP on AWS**: https://registry.opendata.aws/naip/
6. **Planet Academic License**: https://www.planet.com/academic/

---

## 8. Next Actions (Week 1-2 Phase 1)

1. **ALSAT-2B Access**: Send contact requests to ASAL (via LinkedIn/academic network), CNES PEPS, ESA TPM
2. **Sentinel-2 Pipeline**: Implement downloader for 4 Algerian AOIs + 4 global validation AOIs
3. **PROBA-V**: Download MISR dataset from ESA Kelvin
4. **Degradation Model**: Implement ALSAT-2B simulator with estimated parameters
5. **Document**: Update `research_plan.md` if ALSAT-2B access timeline >4 weeks