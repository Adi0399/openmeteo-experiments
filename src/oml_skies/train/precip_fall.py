from __future__ import annotations
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from basketball.data.openmeteo_fetch import fetch_archive_daily
from basketball.features.builders import build_features
from basketball.targets.labels import label_precip_3day_sum

LAT, LON = -33.8678, 151.2073
DAILY = [
    "precipitation_sum","rain_sum","temperature_2m_mean",
    "relative_humidity_2m_mean","surface_pressure_mean","wind_speed_10m_mean"
]

def run(out_dir: str = "models/precipitation_fall"):
    train_start, train_end = "2000-01-01", "2023-12-31"
    valid_start, valid_end = "2024-01-01", "2024-12-31"

    d_train = fetch_archive_daily(LAT, LON, train_start, train_end, DAILY)
    d_valid = fetch_archive_daily(LAT, LON, valid_start, valid_end, DAILY)

    df = pd.concat([d_train, d_valid], ignore_index=True)
    Xall = build_features(df)
    yreg = label_precip_3day_sum(Xall)

    merged = yreg.merge(Xall, on="date", how="left")
    m_train = (merged["date"] < pd.Timestamp("2024-01-01"))
    m_valid = (merged["date"] >= pd.Timestamp("2024-01-01"))

    drop_cols = {"date","start_date","end_date","precip3_sum"}
    feats = [c for c in merged.columns if c not in drop_cols]

    Xtr = merged.loc[m_train, feats].fillna(0)
    ytr = merged.loc[m_train, "precip3_sum"].astype(float)
    Xva = merged.loc[m_valid, feats].fillna(0)
    yva = merged.loc[m_valid, "precip3_sum"].astype(float)

    reg = HistGradientBoostingRegressor(learning_rate=0.06)
    reg.fit(Xtr, ytr)

    yhat = reg.predict(Xva)
    metrics = dict(
        MAE = float(mean_absolute_error(yva, yhat)),
        RMSE = float(np.sqrt(mean_squared_error(yva, yhat))),
        R2 = float(r2_score(yva, yhat)),
        n_train = int(m_train.sum()),
        n_valid = int(m_valid.sum()),
    )
    print(metrics)

    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, f"{out_dir}/model.pkl")
    json.dump({"features": feats}, open(f"{out_dir}/features.json","w"))

if __name__ == "__main__":
    run()
