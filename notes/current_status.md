# Crop Trajectory Research Status

## Current Best Result
v18 benchmark

5-Fold CV:
- Accuracy: 70.9% ± 2.5%
- Balanced Accuracy: 60.0% ± 5.1%
- Macro F1: 60.1% ± 3.7%

## Key Findings
- Dekad data beats monthly data.
- Phenology features help.
- Class weighting helps.
- Wheat vs Barley remains difficult.
- Beans vs Oats improved strongly with dekad resolution.
- Barley dominance causes bias.
- Full balancing reduced performance.
- Heading-date feature experiment did not improve results.

## Next Research
1. Review literature.
2. Add environmental variables:
   - ERA5 temperature
   - ERA5 precipitation
   - SMAP soil moisture
   - DEM features
3. Test S2 + SAR + Environment model.
4. Investigate object/parcel-level classification papers.
