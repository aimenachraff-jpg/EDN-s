# ALSAT-2B Data Access Investigation

## Executive Summary
**Status**: BLOCKED - ASAL website inaccessible, no public data portal found.
**Action Required**: Institutional contact via ASAL, CNES, or ESA channels.

## ALSAT-2 Mission Overview (Verified from ESA eoPortal)
- **Operator**: Algerian Space Agency (ASAL)
- **Satellites**: AlSat-2A (launched 2010), AlSat-2B (launched 2016)
- **Instrument**: NAOMI (New AstroSat Optical Modular Instrument)
- **Orbit**: Sun-synchronous, 670 km altitude, 98.2° inclination, 10:30 LTAN
- **Status**: Operational (extended)

## NAOMI Instrument Specifications (Verified)
| Parameter | Panchromatic (PAN) | Multispectral (MS) |
|-----------|-------------------|-------------------|
| GSD | 2.5 m | 10 m |
| Swath | 17.5 km | 17.5 km |
| Bands | 1 (broadband) | 4 (Blue, Green, Red, NIR) |
| Spectral Range | 0.45 - 0.75 µm | B: 0.45-0.52, G: 0.53-0.60, R: 0.62-0.69, NIR: 0.76-0.89 µm |
| Quantization | 12-bit (10-bit downlink) | 12-bit (10-bit downlink) |
| Detector | 7000 px TDI array | 1750 px per band |
| Telescope | Korsch TMA SiC, 200mm aperture, F/16 | Same |

## Data Access Channels Investigated

### 1. ASAL Official Website (asal.dz)
- **Status**: NOT ACCESSIBLE (connection timeout/refused)
- **Notes**: Multiple attempts failed. May be geo-blocked, down, or migrated.

### 2. ESA eoPortal
- **URL**: https://www.eoportal.org/satellite-missions/alsat-2
- **Data Access**: Metadata only. No direct data download.
- **Contact**: Links to ASAL for data requests.

### 3. CNES (French Space Agency)
- **Role**: Technical partner for ALSAT-2 (Airbus Defence and Space built satellites)
- **Potential Access**: Via CNES catalog or PEPS platform
- **Status**: Not yet contacted

### 4. ESA Earth Online / EO Gateway
- **URL**: https://earth.esa.int/eogateway/
- **Status**: ALSAT-2 page returns 404
- **Notes**: ESA may not host ALSAT-2 data directly

### 5. International Charter / Disaster Management
- **Mechanism**: ALSAT-2 used for disaster monitoring
- **Access**: Via charter activation only (not for research)

## Institutional Contact Strategy

### Primary: ASAL Direct Contact
- **Email**: Need to find current contact (website down)
- **Alternate**: ASAL LinkedIn, Algerian Ministry of Higher Education and Scientific Research
- **Language**: French/Arabic preferred

### Secondary: CNES
- **Contact**: CNES Earth Observation data services
- **Platform**: PEPS (Plateforme d'Exploitation des Produits Sentinel) - may include partner data

### Tertiary: ESA
- **Contact**: ESA Earth Observation Helpdesk
- **Program**: Third Party Missions (TPM) - ALSAT-2 may be registered

## Data Policy Considerations
- **Ownership**: Algerian Government / ASAL
- **Likely Policy**: Restricted - requires research proposal, MoU, or institutional agreement
- **Typical Timeline**: 2-6 months for approval
- **Format**: Likely GeoTIFF, DIMAP, or proprietary

## Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| No data access granted | HIGH | CRITICAL | Prepare substitute datasets immediately |
| Long approval delays (>3 months) | HIGH | HIGH | Start substitute dataset work in parallel |
| Data format incompatibility | MEDIUM | MEDIUM | Budget preprocessing time |
| Limited temporal/spatial coverage | MEDIUM | MEDIUM | Define clear AOI/TOI for request |

## Immediate Actions (This Week)
1. [ ] Find ASAL contact email via LinkedIn/academic networks
2. [ ] Draft formal data request letter (French)
3. [ ] Contact CNES PEPS helpdesk
4. [ ] Contact ESA TPM office
5. [ ] **PARALLEL**: Fully characterize substitute datasets (see dataset_comparison.md)

## Substitute Dataset Readiness
If ALSAT-2B access not granted within 4 weeks, we proceed with:
- **Primary Substitute**: Sentinel-2 (10m MS, public, global, extensive SR literature)
- **Secondary**: PROBA-V MISR (multi-image SR benchmark, 100m-300m)
- **Tertiary**: NAIP (1m, US only), PlanetScope (3m, commercial/academic), WorldView (0.3-0.5m, commercial)

## Decision Gate
**Week 4 of Phase 1**: If no credible path to ALSAT-2B data, formally switch to substitute dataset and update research_plan.md accordingly. Document decision in docs/methodology_decisions.md.