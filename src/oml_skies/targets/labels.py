import pandas as pd

def label_rain_plus_7(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: will it rain (>0 mm) exactly 7 days after date D?
    Uses daily 'rain_sum'. Align future to current with shift(-7).
    """
    df = daily_df.sort_values("date").copy()
    df["will_rain_7d"] = df["rain_sum"].shift(-7).gt(0).astype("Int8")
    df["target_date"] = df["date"] + pd.Timedelta(days=7)
    return df.dropna(subset=["will_rain_7d"])

def label_precip_3day_sum(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Regression label: cumulative precipitation (mm) for D+1..D+3 inclusive.
    Uses daily 'precipitation_sum'. Shift future days back to align with D.
    """
    df = daily_df.sort_values("date").copy()
    fut = (
        df["precipitation_sum"].shift(-1)
        + df["precipitation_sum"].shift(-2)
        + df["precipitation_sum"].shift(-3)
    )
    df["precip3_sum"] = fut.astype("float32")
    df["start_date"] = df["date"] + pd.Timedelta(days=1)
    df["end_date"] = df["date"] + pd.Timedelta(days=3)
    return df.dropna(subset=["precip3_sum"])
