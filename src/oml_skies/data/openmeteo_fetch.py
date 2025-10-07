import requests
import pandas as pd

BASE = "https://archive-api.open-meteo.com/v1/archive"

def fetch_archive_daily(
    lat: float,
    lon: float,
    start: str,
    end: str,
    daily_vars=None,
    timezone: str = "Australia/Sydney",
    precipitation_unit: str = "mm",
) -> pd.DataFrame:
    """
    Fetch daily aggregates from Open-Meteo archive API for [start, end].
    Returns a DataFrame with a 'date' column (datetime64[ns]).
    """
    if daily_vars is None:
        daily_vars = [
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
        ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(daily_vars),
        "timezone": timezone,
        "precipitation_unit": precipitation_unit,
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if "daily" not in js:
        raise ValueError("No 'daily' field in response")
    df = pd.DataFrame(js["daily"])
    df["date"] = pd.to_datetime(df["time"])
    return df
