
## Transfer Learning Results (June 2026)
- BreizhCrops TempCNN pretrained: 89% (French data)
- Fine-tuned on Irish 1294 parcels: 69.9% — beats CatBoost 63.5%
- Converged at epoch 60-100, ceiling at current data representation

## Next Priority: Parcel Statistics (Highest ROI)
Replace centroid 11×11 window with full parcel polygon mask:
  - Sample all pixels inside polygon boundary
  - Compute median + std per band (robust to outliers)
  - Adds within-field variability as a feature
  
Code change needed in extractors/s2_timeseries.py:
  - Use shapely polygon mask on rasterio window
  - Return median AND std per band → [T, 26] instead of [T, 13]

## Priority 2: Sentinel-1 SAR time series
  - Add VV/VH channels alongside S2 bands
  - Target: [T, 15] (13 S2 + 2 SAR bands)
  
## Priority 3: More Irish parcels
  - CDSE resets July 1 → extract 1000+ parcels
  - Target 200+ per minority class (Oats/Beans/OSR)

## Priority 4: Class-weighted loss
  freq = Counter(labels)
  weights = torch.tensor([1/freq[c] for c in le.classes_])
  criterion = nn.NLLLoss(weight=weights)

## CRITICAL FINDING: Temporal Resolution Bottleneck (June 2026)

Binary Beans vs Oats experiment:
  Monthly CatBoost:  52.8%  (near random — information destroyed)
  Dekad CatBoost:    75.0%  (+22.2% uplift)
  TempCNN dekad:     72.2%

CONCLUSION: The production CatBoost model (63.5%) was always
working with destroyed information. Monthly aggregation removes
the harvest timing signal that distinguishes Beans from Oats.

## New Roadmap

1. Retrain full 7-class CatBoost on FLAT DEKAD features [N, 585]
   instead of monthly [N, 48]
   Expected: 70%+ overall accuracy

2. Retrain TempCNN fusion on dekad S2 + dekad SAR
   (need to extract SAR at dekad resolution too)
   Expected: 78%+

3. The 1294 S2 dekad parcels already extracted are the right format
   Just need SAR at dekad resolution for the 420 overlap parcels

## Key insight
Not more data. Not better model.
FINER TEMPORAL RESOLUTION around harvest window (Jul-Aug).

## Planque et al. 2021 — Full Algorithm Decoded

Key discriminating feature (Barley vs Wheat):
  NOT magnitude — TIMING of VH peak relative to VH/VV peak

  Barley: VH peak COINCIDES with VH/VV peak
          (awns cause simultaneous biomass + ear signal)
  Wheat:  VH peak AFTER VH/VV peak
          (ear emerges after flag leaf = after max biomass)

Variables needed (all from Sentinel-1):
  VH, VV, VH/VV ratio — parcel median per acquisition date

Statistical tests:
  Mann-Kendall: detects significant trend direction
  Sen's slope:  quantifies slope magnitude (Ssl++, Ssl+, Ssl−, Ssl−−)
  p-value threshold: 0.01

Algorithm window periods for Ireland:
  Winter separation:  Nov-Mar (VH/VV stable for spring crops)
  Stem elongation:    Apr-May
  Barley ifl window:  May-Jun  (dekads 13-18)
  Wheat ifl window:   Jun-Jul  (dekads 16-21)

Irish data requirements July 1:
  SAR at 6-12 day resolution (NOT monthly)
  CDSE: extract all ascending + descending passes per parcel
  Target: 24+ SAR dates across growing season
