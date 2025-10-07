import pandas as pd
import numpy as np

ROLLS = [3, 7, 14, 28]
NUM_COLS = [
    "precipitation_sum",
    "rain_sum",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_mean",
]

def add_rolls(df: pd.DataFrame, cols=NUM_COLS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for w in ROLLS:
            out[f"{c}_mean_{w}"] = out[c].rolling(w, min_periods=max(2, w//2)).mean()
            out[f"{c}_sum_{w}"]  = out[c].rolling(w, min_periods=max(2, w//2)).sum()
    return out

def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    out["cos_doy"] = np.cos(2*np.pi*doy/365.25)
    out["month"] = out["date"].dt.month.astype("int16")
    return out

def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df[["date"] + NUM_COLS].copy()
    df = add_rolls(df)
    df = add_seasonality(df)
    return df
