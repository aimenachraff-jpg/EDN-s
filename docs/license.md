# Dataset Licenses & Attribution

**Last updated:** 2026-11-13  
**Verification sources:** Zenodo API (PROBA-V), Copernicus official policy pages, Microsoft Planetary Computer terms

---

## PROBA-V Super-Resolution Dataset

### License
**Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Zenodo record: `https://zenodo.org/doi/10.5281/zenodo.6327426`
- License ID: `cc-by-4.0` (confirmed via Zenodo API metadata.license.id)
- Official dataset title: "PROBA-V Super-Resolution dataset" (not "Super-human Verification Challenge")

### Required Attribution
When using this dataset in any publication or derivative work, cite:

> Märtens M., Izzo D., Krzic A. and Cox D. "Super-resolution of PROBA-V images using convolutional neural networks." Astrodynamics 3.4 (2019): 387-402. ([arXiv:1907.01821](https://arxiv.org/abs/1907.01821))

And include the Zenodo DOI as a data reference:
> Märtens, M., Izzo, D., Krzic, A., & Cox, D. (2018). PROBA-V Super-Resolution dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6327426

### Usage Conditions
Under CC BY 4.0 you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

The only condition is **attribution**: appropriate credit must be given to the original authors and source.

### Source
- ESA Kelvins: `https://kelvins.esa.int/proba-v-super-resolution/data/`
- Zenodo: `https://zenodo.org/record/6327426`
- Download URL: `https://zenodo.org/api/records/6327426/files/probav_data.zip/content`
- File size: 708,416,603 bytes (675 MB)
- MD5 checksum: `7101101c7c91eeb5979d472861c1b784`
- 74 hand-selected regions, 1,450 total scenes (1,160 train / 290 test)

---

## Sentinel-2 (Copernicus / Microsoft Planetary Computer)

### License
**Open Access — Free under CC BY 4.0** (Copernicus Open Access Policy)

Per Copernicus official policy ([eu-space.europa.eu](https://eu-space.europa.eu/programmes/earth-observation-copernicus)):
> "The vast majority of data/information delivered by Copernicus is made available and accessible to any citizen, and any organisation around the world **on a free, full, and open basis**."

The Copernicus data policy specifies CC BY 4.0 for Sentinel data products. Microsoft Planetary Computer (`https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a`) provides Sentinel-2 Level-2A Surface Reflectance under the same terms.

### Citation
European Space Agency (ESA), Copernicus Programme (2015–ongoing). Sentinel-2 User Handbook.

### Required Attribution for This Project
> "Sentinel-2 Level-2A surface reflectance. European Space Agency."

### Source for This Project
- **Primary:** Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`)
- **Collection:** `sentinel-2-l2a` (Level-2A Surface Reflectance)
- **Bands acquired:** B2 (Blue 490 nm), B3 (Green 560 nm), B4 (Red 665 nm), B8 (NIR 842 nm)
- **Processing level:** L2A (surface reflectance, values in [0, 1])
- **AOIs:** 8 planned regions (DZ_01–DZ_04, FR_01, US_01, CL_01, BR_01)

---

## Notes on Third-Party Benchmarks Used

| Dataset / Benchmark | License | Citation Required |
|---|---|---|
| PROBA-V Super-Resolution | CC BY 4.0 | Yes (Märtens et al. 2019) |
| EuroSAT | CC BY 4.0 | Yes (Helber et al. 2019) |
| OSMBuildings | ODbL 1.0 | Yes (openstreetmap.org) |
| OSCD (Change Detection) | CC BY 4.0 | Yes (Diakaki et al. 2020) |
| LandCover.ai | CC BY-SA 4.0 | Yes |

---

## Code License
**MIT License** — Free use, modification, distribution with attribution

---

*This document is part of the MASTER2-A1 thesis project. All licenses are subject to their current versions as published by the respective rights holders. Verification performed 2026-11-13 against official APIs and policy pages.*
