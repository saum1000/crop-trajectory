
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
