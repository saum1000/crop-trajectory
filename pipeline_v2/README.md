# Crop Classification Pipeline v2
## Clean implementation — June 2026

### Data Plan:
- Labels: DAFM 2025 declarations
- Sentinel-2: 2025 full season (Jan-Sep)
- Sentinel-1 SAR: 2025 full season via HyP3
- ERA5: 2025 weather

### Architecture:
- Hierarchical binary classifiers (cascade)
- CatBoost per stage
- Stage 1: OSR vs rest
- Stage 2: Maize vs rest
- Stage 3: Barley vs Wheat
- Stage 4: Oats vs Beans

### Target: 70%+ balanced accuracy
