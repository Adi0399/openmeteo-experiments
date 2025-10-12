# Open Meteo — Sydney Rain & Precipitation ML (Experiments)

This repository contains the experimentation code for two models using Open-Meteo historical data for Sydney (lat -33.8678, lon 151.2073):

1) **Rain +7 days (binary):** will it rain exactly 7 days after the input date?
2) **3-day cumulative precipitation (regression):** total precipitation (mm) for D+1..D+3.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
Train models
bash
Copy code
# Rain +7 days (saves to models/rain_or_not/)
python -m oml_skies.train.rain_or_not

# 3-day precip (saves to models/precipitation_fall/)
python -m oml_skies.train.precip_fall
Package modules (local)
src/oml_skies/data/openmeteo_fetch.py

src/oml_skies/features/builders.py

src/oml_skies/targets/labels.py

src/oml_skies/train/*.py

Notebooks
Follow the naming convention:
notebooks/
├─ rain_or_not/
│ └─ 36120-25SP-<25410263_id>-experiment_1.ipynb
└─ precipitation_fall/
└─ 36120-25SP-<25410263>-experiment_2.ipynb

Artifacts
models/
├─ rain_or_not/{model.pkl, features.json}
└─ precipitation_fall/{model.pkl, features.json}

Notes
2000–2023 train, 2024 validation, 2025+ = production only.

No future leakage: features only up to the input date D.
