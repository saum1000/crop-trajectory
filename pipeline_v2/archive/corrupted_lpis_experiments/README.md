# CORRUPTED EXPERIMENTS — DO NOT USE FOR SCIENTIFIC CONCLUSIONS

## Date discovered: June 22 2026

## Root cause
The DAFM GeoAPI does not support offset-based pagination.
Every call with offset=N returns the same first page regardless of N.

Proof:
  OFFSET 0:    679943CC0D14, B8F562998F48, 1207A0BDAFF8
  OFFSET 1000: 679943CC0D14, B8F562998F48, 1207A0BDAFF8
  OFFSET 5000: 679943CC0D14, B8F562998F48, 1207A0BDAFF8

## Consequence
Downloads that appeared to collect 5000-5120 parcels
were actually downloading the same ~157 unique parcels
repeated 20-32 times each.

Evidence:
  2024 LPIS claimed: 5120 rows
  2024 LPIS actual:  157 unique PAR_LAB values
  2022 LPIS claimed: 5000 rows  
  2022 LPIS actual:  245 unique PAR_LAB values

## Results produced (INVALID)
All results from these datasets are invalid:

  2022+2023+2024 → 2025: 95-98% balanced accuracy
  2022+2023+2025 → 2024: 85% balanced accuracy
  Same-year CV:           99% balanced accuracy

These results appeared valid because:
  - Shuffled label test passed (29% shuffled vs 80% real)
  - Inter-year correlation was low (-0.158)
  - Parcel ID overlap was zero

But the underlying data was 157 unique fields repeated
32 times. The fid split created train/test from the
SAME physical fields under different repetition indices.

## Fix applied
Replaced offset-based pagination with tile-based download:
  70 tiles of 0.1°×0.1° covering Carlow/Wexford
  Result: 4865 truly unique parcels, 4865 unique PAR_LABs
  File: pipeline_v2/data/labels/lpis_2024_unique.json

## GEE assets (also invalid, archived)
  projects/ireland-mrv-prototype/assets/s2_2022_carlow
  projects/ireland-mrv-prototype/assets/s2_2023_carlow
  projects/ireland-mrv-prototype/assets/s2_2024_carlow
  projects/ireland-mrv-prototype/assets/s2_2025_carlow

New clean assets will be:
  projects/ireland-mrv-prototype/assets/s2_2022_clean
  projects/ireland-mrv-prototype/assets/s2_2023_clean
  projects/ireland-mrv-prototype/assets/s2_2024_clean
  projects/ireland-mrv-prototype/assets/s2_2025_clean
