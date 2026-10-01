"""
src/ml/predict_risk.py
Phase 9: Real-time road risk inference engine with physical seismic attenuation.
"""

import joblib
import numpy as np
import pandas as pd
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
        threshold: float = 0.75,
    ) -> pd.DataFrame:
        df = self.roads_df.copy()

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
        raw_probs = self.model.predict_proba(X)[:, 1]

        # Ground motion physical attenuation for regional / far-field events
        # Events > 250km decay significantly before reaching Dhaka
        dist_km = df["epicentral_dist_km"].values
        attenuation_factor = np.where(
            dist_km > 250.0,
            np.exp(-0.007 * (dist_km - 250.0)),
            1.0
        )
        
        calibrated_probs = np.clip(raw_probs * attenuation_factor, 0.02, 0.98)

        df["risk_prob"] = calibrated_probs.round(4)
        df["is_blocked"] = (df["risk_prob"] >= threshold).astype(int)

        return df