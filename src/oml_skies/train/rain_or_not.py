from __future__ import annotations
import json
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
)

from basketball.data.openmeteo_fetch import fetch_archive_daily
from basketball.features.builders import build_features
from basketball.targets.labels import label_rain_plus_7

LAT, LON = -33.8678, 151.2073
DAILY = [
    "precipitation_sum","rain_sum","temperature_2m_mean",
    "relative_humidity_2m_mean","surface_pressure_mean","wind_speed_10m_mean"
]

def run(out_dir: str = "models/rain_or_not"):
    train_start, train_end = "2000-01-01", "2023-12-31"
    valid_start, valid_end = "2024-01-01", "2024-12-31"

    d_train = fetch_archive_daily(LAT, LON, train_start, train_end, DAILY)
    d_valid = fetch_archive_daily(LAT, LON, valid_start, valid_end, DAILY)

    df = pd.concat([d_train, d_valid], ignore_index=True)
    Xall = build_features(df)
    ybin = label_rain_plus_7(Xall)

    merged = ybin.merge(Xall, on="date", how="left")
    m_train = (merged["date"] < pd.Timestamp("2024-01-01"))
    m_valid = (merged["date"] >= pd.Timestamp("2024-01-01"))

    drop_cols = {"date","target_date","will_rain_7d"}
    feats = [c for c in merged.columns if c not in drop_cols]

    Xtr, ytr = merged.loc[m_train, feats].fillna(0), merged.loc[m_train, "will_rain_7d"].astype(int)
    Xva, yva = merged.loc[m_valid, feats].fillna(0), merged.loc[m_valid, "will_rain_7d"].astype(int)

    clf = HistGradientBoostingClassifier(learning_rate=0.06)
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xva)[:,1]
    metrics = dict(
        pr_auc = float(average_precision_score(yva, p)),
        roc_auc = float(roc_auc_score(yva, p)),
        f1 = float(f1_score(yva, (p>=0.5).astype(int))),
        balanced_accuracy = float(balanced_accuracy_score(yva, (p>=0.5).astype(int))),
        n_train = int(m_train.sum()),
        n_valid = int(m_valid.sum()),
    )
    print(metrics)

    import os, pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, f"{out_dir}/model.pkl")
    json.dump({"features": feats}, open(f"{out_dir}/features.json","w"))

if __name__ == "__main__":
    run()
