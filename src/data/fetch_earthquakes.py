"""
src/data/fetch_earthquakes.py
Fetches historical seismic events around Dhaka from the USGS API.
"""

import os
from datetime import datetime
import pandas as pd
import requests

USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Study parameters
DHAKA_LAT = 23.8103
DHAKA_LON = 90.4125
MAX_RADIUS_KM = 800.0
MIN_MAGNITUDE = 4.5
START_TIME = "1990-01-01"


def fetch_earthquake_data(
    lat: float = DHAKA_LAT,
    lon: float = DHAKA_LON,
    radius_km: float = MAX_RADIUS_KM,
    min_mag: float = MIN_MAGNITUDE,
    start_date: str = START_TIME,
    output_path: str = "data/raw/earthquakes.csv",
) -> pd.DataFrame:
    """Queries USGS Earthquake API and writes standardized CSV."""
    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "starttime": start_date,
        "endtime": datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
        "orderby": "time-asc",
    }

    print(
        f"[INFO] Requesting earthquakes: Lat={lat}, Lon={lon}, "
        f"Radius={radius_km}km, MinMag={min_mag}..."
    )

    response = requests.get(USGS_API_URL, params=params, timeout=30)
    response.raise_for_status()

    geojson = response.json()
    features = geojson.get("features", [])
    print(f"[INFO] Downloaded {len(features)} seismic events from USGS.")

    records = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        records.append(
            {
                "event_id": f.get("id"),
                "time": pd.to_datetime(props.get("time"), unit="ms"),
                "latitude": geom["coordinates"][1],
                "longitude": geom["coordinates"][0],
                "depth_km": geom["coordinates"][2],
                "magnitude": props.get("mag"),
                "magnitude_type": props.get("magType"),
                "place": props.get("place"),
                "status": props.get("status"),
            }
        )

    df = pd.DataFrame(records)
    # Deduplicate and sort chronologically
    df = df.drop_duplicates(subset=["event_id"]).sort_values("time").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {len(df)} verified events to '{output_path}'.")
    return df


if __name__ == "__main__":
    df_quakes = fetch_earthquake_data()
    print("\n--- Summary Statistics ---")
    print(df_quakes[["magnitude", "depth_km"]].describe())
    print("\nSample records:")
    print(df_quakes.head(3))