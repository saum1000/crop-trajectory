# Crop Classification Results — Cube Earth Pipeline v2

## Date
June 2026

## Dataset
- **Parcels:** 4865 unique LPIS 2024 arable parcels (Carlow/Wexford/Kilkenny)
- **Years:** 2022, 2023, 2024, 2025
- **Source:** Sentinel-2 SR Harmonized (GEE)
- **Cloud masking:** SCL pixel-level masking (excludes cloud, shadow, cirrus)
- **Compositing:** 10-day dekad median composite
- **Gap filling:** Linear interpolation per parcel per metric
- **Features:** NDVI, NDRE, EVI, NDWI, NDII × 32 dekads × 3 training years
- **Labels:** DAFM LPIS 2024 crop declarations

## Key Data Quality Notes
- Previous pipeline used scene-level cloud filter (<85%) → 0.7-0.8 NDVI error
- SCL masking reduces this to near-zero for clear pixels
- 2022 had 31.4% missing dekads (cloud) → weakest year
- 2023-2025 had 19-38% missing, filled by interpolation
- LPIS API pagination was broken (offset ignored) → fixed with tile-based download

## Results

### 10-Class Classification (all crops)
Train: 2022+2023+2024 → Test: 2025 (unseen parcels, different year)

| Features | Balanced Accuracy | Macro F1 |
|----------|------------------|----------|
| NDVI only | 64.3% | 61.2% |
| NDVI+NDRE | 68.1% | 62.9% |
| NDVI+NDRE+EVI | 66.9% | 63.7% |
| NDVI+NDRE+EVI+NDWI | 68.3% | 63.9% |
| All 5 indices | **70.3%** | **66.7%** |

#### Per-class (all 5 indices):
| Crop | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| Spring Barley | 0.96 | 0.75 | 0.84 | 584 |
| Winter Barley | 0.90 | 0.90 | 0.90 | 86 |
| Winter Wheat | 0.78 | 0.90 | 0.84 | 82 |
| Maize | 0.68 | 0.97 | 0.80 | 67 |
| OSR | 0.92 | 0.92 | 0.92 | 36 |
| Beans | 0.67 | 0.83 | 0.74 | 35 |
| Potatoes | 0.64 | 0.47 | 0.54 | 15 |
| Spring Oats | 0.18 | 0.64 | 0.28 | 39 |
| Spring Wheat | 0.00 | 0.00 | 0.00 | 12 |
| Winter Oats | 0.82 | 0.53 | 0.64 | 17 |

### 5-Class Classification (major crops)
Spring Barley, Winter Barley, Winter Wheat, Maize, OSR

| Metric | Value |
|--------|-------|
| **Balanced Accuracy** | **91.7%** |
| **Overall Accuracy** | **93%** |
| Macro F1 | 89% |

| Crop | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Spring Barley | 0.71 | 0.97 | 0.82 |
| Winter Barley | 0.94 | 0.89 | 0.91 |
| Winter Wheat | 0.98 | 0.93 | 0.96 |
| Maize | 0.91 | 0.90 | 0.90 |
| OSR | 0.81 | 0.90 | 0.86 |

### Temporal Transfer Learning
| Training years | Test year | Balanced Accuracy |
|----------------|-----------|------------------|
| 1 year | next year | ~15% |
| 2 years (2023+2024) | 2025 | 60.6% |
| 3 years (2022+2023+2024) | 2025 | 70.3% |
| Same year (ceiling) | same | 65% avg |

## Conclusions
1. **Major crop classification is commercially viable** at 91.7% balanced accuracy
2. **3 years of training data is sufficient** for cross-year generalization
3. **SCL pixel masking is essential** — scene-level filtering fails in Ireland
4. **Minority classes need more data**: Spring Oats, Spring Wheat, Potatoes
5. **Next steps**: Add Sentinel-1 SAR, build production inference, add confidence scores

## Files
- Labels: `pipeline_v2/data/labels/lpis_2024_unique.json`
- Dataset: `pipeline_v2/data/s2/multiyear_scl_2022_2025.csv`
- GEE assets: `projects/ireland-mrv-prototype/assets/s2_{year}_scl_b{batch}`
- Corrupted experiments archived: `pipeline_v2/archive/corrupted_lpis_experiments/`
