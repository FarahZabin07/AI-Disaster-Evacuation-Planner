"""
main.py
-------
CLI entry point for the real-data AI Disaster Evacuation Planner (Dhaka).
Runs the end-to-end pipeline:
1. Loads real Dhaka arterial road network.
2. Accepts or simulates an earthquake event (USGS style).
3. Predicts real-time road blockage risk via Logistic Regression.
4. Dynamically annotates the road network graph.
5. Reports road hazard and connectivity statistics.

Usage:
    python main.py
    python main.py --mag 6.0 --lat 25.18 --lon 91.80 --depth 35.0 --threshold 0.80
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd

from src.ml.predict_risk import RoadRiskPredictor
from src.graph.build_graph import load_base_graph, annotate_graph_with_risk

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Based Disaster Evacuation Planner for Dhaka - CLI Runner"
    )
    # Default scenario: Regional Mw 6.0 event in Dauki Fault zone (~210km from Dhaka)
    parser.add_argument("--lat", type=float, default=25.18, help="Earthquake epicenter latitude")
    parser.add_argument("--lon", type=float, default=91.80, help="Earthquake epicenter longitude")
    parser.add_argument("--mag", type=float, default=6.0, help="Earthquake magnitude (Mw)")
    parser.add_argument("--depth", type=float, default=35.0, help="Hypocentral depth in km")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Probability threshold to flag a road as severely blocked (0.0 to 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print(" AI-BASED DISASTER EVACUATION PLANNER FOR DHAKA (REAL DATA PIPELINE)")
    print("=" * 70)

    print("\n[1] SEISMIC EVENT PARAMETERS:")
    print(f"    Magnitude:     Mw {args.mag}")
    print(f"    Epicenter:     Lat {args.lat:.4f}, Lon {args.lon:.4f}")
    print(f"    Depth:         {args.depth:.1f} km")
    print(f"    Block Cutoff:  P(risk) >= {args.threshold:.2f}")

    # Step 1: Real-time ML Inference
    print("\n[2] RUNNING LOGISTIC REGRESSION ROAD HAZARD INFERENCE...")
    try:
        predictor = RoadRiskPredictor()
        risk_df = predictor.predict_for_earthquake(
            eq_lat=args.lat,
            eq_lon=args.lon,
            magnitude=args.mag,
            depth_km=args.depth,
            threshold=args.threshold,
        )
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        return 1

    total_roads = len(risk_df)
    blocked_count = int(risk_df["is_blocked"].sum())
    high_risk_open = int(((risk_df["risk_prob"] >= 0.40) & (risk_df["risk_prob"] < args.threshold)).sum())
    safe_roads = total_roads - blocked_count - high_risk_open

    print(f"\n[3] ROAD HAZARD INFERENCE SUMMARY:")
    print(f"    Total Arterial Segments: {total_roads:,}")
    print(f"    Severely Blocked (Cut):  {blocked_count:,} ({blocked_count / total_roads * 100:.1f}%)")
    print(f"    High-Risk but Passable:  {high_risk_open:,} ({high_risk_open / total_roads * 100:.1f}%)")
    print(f"    Safe / Low-Risk Roads:   {safe_roads:,} ({safe_roads / total_roads * 100:.1f}%)")

    print("\n    Top 5 Most Vulnerable Dhaka Road Corridors:")
    cols_show = ["road_id", "highway_clean", "epicentral_dist_km", "risk_prob", "is_blocked"]
    top_vulnerable = risk_df.sort_values("risk_prob", ascending=False).head(5)
    print(top_vulnerable[cols_show].to_string(index=False))

    # Step 2: Dynamic Graph Annotation
    print("\n[4] ANNOTATING DHAKA ROAD NETWORK GRAPH...")
    try:
        base_G = load_base_graph()
        risk_G = annotate_graph_with_risk(
            base_G, risk_df, blockage_threshold=args.threshold
        )
    except Exception as exc:
        print(f"[ERROR] Graph annotation failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print(" STATUS: Seismic inference and dynamic road graph successfully built.")
    print(" Ready for Phase 11-13 (A* Pathfinding) & Phase 14-15 (CSP Shelter Allocation).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())