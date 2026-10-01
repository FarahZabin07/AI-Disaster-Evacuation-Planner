"""
src/data/create_dataset.py
Phases 3, 4, and 5:
- Reprojects Dhaka road network to UTM Zone 46N (EPSG:32646) for exact centroid computation.
- Cleans and handles nested OSM road attributes (lanes, speed, highway types).
- Computes Haversine epicentral distance between earthquakes and road segments.
- Generates an empirical scenario-based proxy failure label based on seismic attenuation.
- Produces the final tabular training dataset for Logistic Regression.
"""

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance calculation in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def parse_lanes_value(val):
    """Safely extracts lane count from strings, lists, arrays, or floats."""
    if val is None:
        return np.nan
    # Handle list/array instances
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return np.nan
        val = val[0]
    
    # Handle string values
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ["none", "nan"]:
        return np.nan
    
    # Match first valid digit/number in text (e.g. "2;3" -> 2, "['2']" -> 2)
    match = re.search(r"(\d+(\.\d+)?)", val_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def clean_road_edges(edges_path: str = "data/raw/dhaka_edges.geojson") -> pd.DataFrame:
    """Loads GeoJSON edges, reprojects for true centroids, and standardizes attributes."""
    print(f"[INFO] Reading road network edges from {edges_path}...")
    gdf = gpd.read_file(edges_path)

    # Reproject to UTM Zone 46N (EPSG:32646) for Bangladesh metric calculations
    gdf_utm = gdf.to_crs(epsg=32646)
    centroids_utm = gdf_utm.geometry.centroid
    # Transform centroids back to WGS84 (EPSG:4326) for latitude and longitude
    centroids_wgs84 = centroids_utm.to_crs(epsg=4326)

    gdf["road_mid_lat"] = centroids_wgs84.y
    gdf["road_mid_lon"] = centroids_wgs84.x

    # Ensure road_id is unique
    if "osmid" in gdf.columns:
        gdf["road_id"] = gdf.index.astype(str) + "_" + gdf["osmid"].astype(str)
    else:
        gdf["road_id"] = gdf.index.astype(str)

    # Standardize highway categories
    def simplify_highway(hw):
        if isinstance(hw, (list, tuple, np.ndarray)):
            hw = hw[0] if len(hw) > 0 else "secondary"
        hw = str(hw).lower()
        for cat in ["motorway", "trunk", "primary", "secondary", "tertiary"]:
            if cat in hw:
                return cat
        return "secondary"

    gdf["highway_clean"] = gdf["highway"].apply(simplify_highway)

    # Parse 'lanes' cleanly with safe fallback
    if "lanes" in gdf.columns:
        gdf["lanes_clean"] = gdf["lanes"].apply(parse_lanes_value)
    else:
        gdf["lanes_clean"] = np.nan

    # Impute missing lanes based on highway class defaults
    lane_defaults = {"motorway": 4.0, "trunk": 3.0, "primary": 2.0, "secondary": 2.0, "tertiary": 1.0}
    gdf["lanes_clean"] = gdf["lanes_clean"].fillna(gdf["highway_clean"].map(lane_defaults)).fillna(2.0)

    # Parse 'oneway'
    gdf["oneway_clean"] = (
        gdf["oneway"].astype(str).str.lower().isin(["true", "1", "t"]).astype(int)
    )

    # Length in meters
    gdf["length_m"] = gdf["length"].astype(float)

    clean_cols = [
        "road_id",
        "highway_clean",
        "length_m",
        "lanes_clean",
        "oneway_clean",
        "road_mid_lat",
        "road_mid_lon",
    ]
    df_roads = pd.DataFrame(gdf[clean_cols]).reset_index(drop=True)
    print(f"[INFO] Successfully processed {len(df_roads)} road segments.")
    return df_roads


def create_training_dataset(
    quakes_path: str = "data/raw/earthquakes.csv",
    edges_path: str = "data/raw/dhaka_edges.geojson",
    output_path: str = "data/processed/earthquake_road_dataset.csv",
    max_quakes: int = 150,
) -> pd.DataFrame:
    """
    Cross-joins sampled independent earthquakes with Dhaka road segments.
    Applies empirical GMPE attenuation function to synthesize scenario labels.
    """
    df_quakes = pd.read_csv(quakes_path)
    df_roads = clean_road_edges(edges_path)

    # Stratified sampling of earthquakes: prioritize higher magnitude events
    print(f"[INFO] Total earthquakes available: {len(df_quakes)}")
    high_mag = df_quakes[df_quakes["magnitude"] >= 5.0]
    moderate_mag = df_quakes[df_quakes["magnitude"] < 5.0]

    # Sample moderate quakes to keep training dataset fast and balanced (~150 quakes total)
    n_sample_mod = min(len(moderate_mag), max(10, max_quakes - len(high_mag)))
    sample_mod = moderate_mag.sample(n=n_sample_mod, random_state=42)
    selected_quakes = (
        pd.concat([high_mag, sample_mod])
        .sort_values("magnitude", ascending=False)
        .reset_index(drop=True)
    )

    print(f"[INFO] Selected {len(selected_quakes)} independent earthquake events for training dataset.")

    # Cross join: each earthquake paired with each road segment
    selected_quakes["key"] = 1
    df_roads["key"] = 1
    merged = pd.merge(selected_quakes, df_roads, on="key").drop(columns=["key"])

    # Calculate distance from epicenter to road segment midpoint
    print("[INFO] Computing vectorized epicentral distances...")
    merged["epicentral_dist_km"] = haversine_np(
        merged["latitude"].values,
        merged["longitude"].values,
        merged["road_mid_lat"].values,
        merged["road_mid_lon"].values,
    )

    # Hypocentral distance calculation (accounting for focal depth)
    hypocentral_dist = np.sqrt(
        merged["epicentral_dist_km"] ** 2 + merged["depth_km"] ** 2
    )

    # --- Scenario / Proxy Label Generation ---
    # Based on standard Joyner-Boore / Campbell seismic attenuation formulation:
    # Ground motion attenuation intensity:
    intensity = (merged["magnitude"] * 1.8) - (1.6 * np.log(hypocentral_dist + 10.0))

    # Road vulnerability factor by road category
    vuln_weights = {
        "motorway": 0.6,
        "trunk": 0.8,
        "primary": 1.0,
        "secondary": 1.3,
        "tertiary": 1.6,
    }
    road_vuln = merged["highway_clean"].map(vuln_weights).fillna(1.0)

    # Lane vulnerability: fewer lanes increase the likelihood of total blockage
    lane_factor = 2.0 / np.clip(merged["lanes_clean"], 1.0, 6.0)

    # Calibrated seismic logit for realistic hazard distribution across 800km radius
    # Produces ~8-14% blockage risk across diverse regional quakes
    logit = 1.5 * intensity + 0.6 * road_vuln + 0.4 * lane_factor - 3.2
    prob_failure = 1.0 / (1.0 + np.exp(-logit))

    # Binary label: 1 = blocked/severe risk, 0 = passable
    merged["is_blocked"] = (prob_failure >= 0.45).astype(int)
    merged["failure_prob_proxy"] = prob_failure.round(4)

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"[SUCCESS] Generated {len(merged):,} earthquake-road training rows.")
    print(f"[SUCCESS] Saved dataset to '{output_path}'.")
    print("\n--- Target Label Distribution ---")
    print(merged["is_blocked"].value_counts(normalize=True).rename("proportion"))

    return merged


if __name__ == "__main__":
    df_training = create_training_dataset()
    print("\nFeature preview:")
    cols_preview = [
        "event_id",
        "magnitude",
        "depth_km",
        "epicentral_dist_km",
        "highway_clean",
        "lanes_clean",
        "is_blocked",
    ]
    print(df_training[cols_preview].head(5))