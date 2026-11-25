# CareLoop Clinics — No-Show Guard

## Results
- Cleaned appointments: 990 rows (from 1020 raw, 32 rejected)
- No-shows: 218 (22.0%)
- Best model: Logistic Regression
- ROC-AUC: 0.77 (cross-validated)
- Threshold used: 0.45
- Test set: 198 rows, 49 flagged as high-risk

## Files
- 01_ingest_clean.ipynb - Part 1: Cleaning
- schema.sql / load_data.sql - Part 2: SQL
- 03_ml_model.ipynb - Part 3: ML Model
- 04_gemini.ipynb - Part 4: GenAI notes analysis

## Limitations
- Only 10 call notes analysed, cannot show real clinic reasons
- Model trained on Jan-Jun 2025 data only
- Weather data is for clinic city, not patient location
