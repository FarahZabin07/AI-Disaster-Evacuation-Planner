"""
src/ml/predict_risk.py
Phase 9: Real-time road risk inference engine.
Takes an earthquake event (lat, lon, mag, depth), loads the trained Logistic Regression
model, and outputs risk probabilities and blocked statuses for all Dhaka road segments.
"""

import joblib
import pandas as pd
import geopandas as gpd
from src.data.create_dataset import haversine_np, clean_road_edges

MODEL_PATH = "models/road_risk_logistic_model.joblib"
EDGES_PATH = "data/raw/dhaka_edges.geojson"


class RoadRiskPredictor:
    def __init__(self, model_path: str = MODEL_PATH, edges_path: str = EDGES_PATH):
        print(f"[INFO] Initializing RoadRiskPredictor with model '{model_path}'...")
        self.model = joblib.load(model_path)
        self.roads_df = clean_road_edges(edges_path)

    def predict_for_earthquake(
        self,
        eq_lat: float,
        eq_lon: float,
        magnitude: float,
        depth_km: float,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Calculates dynamic road failure probabilities across the road network
        for a specified seismic scenario.
        """
        df = self.roads_df.copy()

        # Compute dynamic seismic features for this specific event
        df["magnitude"] = magnitude
        df["depth_km"] = depth_km
        df["epicentral_dist_km"] = haversine_np(
            eq_lat, eq_lon, df["road_mid_lat"].values, df["road_mid_lon"].values
        )

        feature_cols = [
            "magnitude",
            "depth_km",
            "epicentral_dist_km",
            "length_m",
            "lanes_clean",
            "highway_clean",
            "oneway_clean",
        ]

        X = df[feature_cols]

        # Predict continuous probability of blockage
        probs = self.model.predict_proba(X)[:, 1]
        df["risk_prob"] = probs.round(4)
        df["is_blocked"] = (probs >= threshold).astype(int)

        return df


if __name__ == "__main__":
    # Test case: Severe Madhupur Fault seismic event near Dhaka
    # Lat: 24.10, Lon: 90.25 (approx 35km North-West of Dhaka), M6.8, Depth: 12km
    predictor = RoadRiskPredictor()
    results = predictor.predict_for_earthquake(
        eq_lat=24.10, eq_lon=90.25, magnitude=6.8, depth_km=12.0
    )

    print("\n--- Inference Simulation Result ---")
    print(f"Total Roads Evaluated: {len(results):,}")
    print(f"Roads Flagged Blocked: {results['is_blocked'].sum():,} ({results['is_blocked'].mean()*100:.1f}%)")
    print("\nSample Risk Table:")
    print(
        results[
            ["road_id", "highway_clean", "epicentral_dist_km", "risk_prob", "is_blocked"]
        ].head(5)
    )