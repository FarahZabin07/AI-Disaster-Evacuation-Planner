"""
src/graph/build_graph.py
Phase 10: Converts raw OSMnx graph into a risk-annotated evacuation graph.
Attaches risk probabilities, blockage flags, and risk-penalized edge weights.
"""

import os
import networkx as nx
import osmnx as ox
import pandas as pd

GRAPH_RAW_PATH = "data/raw/dhaka_network.graphml"


def load_base_graph(graph_path: str = GRAPH_RAW_PATH) -> nx.MultiDiGraph:
    """Loads Dhaka GraphML topology."""
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Base graph not found at {graph_path}. Run Phase 2 first.")
    print(f"[INFO] Loading raw network graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    return G


def annotate_graph_with_risk(
    G: nx.MultiDiGraph,
    risk_df: pd.DataFrame,
    risk_weight_factor: float = 6.0,
    blockage_threshold: float = 0.50,
) -> nx.MultiDiGraph:
    """
    Annotates graph edges with predicted risk and computes dynamic routing cost.
    Edges with risk >= blockage_threshold are marked non-traversable (blocked).
    Traversable edges receive cost = length * (1.0 + risk_weight_factor * risk_prob^2).
    """
    risk_map = risk_df.set_index("road_id")[["risk_prob"]].to_dict("index")

    annotated_count = 0
    blocked_count = 0
    default_assigned_count = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_id = f"{u}_{v}_{key}"
        length = float(data.get("length", 100.0))

        if edge_id in risk_map:
            risk_prob = float(risk_map[edge_id]["risk_prob"])
        else:
            rev_edge_id = f"{v}_{u}_{key}"
            if rev_edge_id in risk_map:
                risk_prob = float(risk_map[rev_edge_id]["risk_prob"])
            else:
                risk_prob = 0.05
                default_assigned_count += 1

        is_blocked = (risk_prob >= blockage_threshold)

        data["risk_prob"] = round(risk_prob, 4)
        data["is_blocked"] = bool(is_blocked)

        if is_blocked:
            blocked_count += 1
            data["evacuation_cost"] = float("inf")
            data["traversable"] = False
        else:
            data["evacuation_cost"] = length * (1.0 + (risk_weight_factor * (risk_prob ** 2)))
            data["traversable"] = True

        annotated_count += 1

    print(f"[INFO] Successfully annotated {annotated_count} graph edges.")
    if default_assigned_count > 0:
        print(f"[INFO] Default baseline assigned: {default_assigned_count} edges.")
    print(f"[INFO] Graph edges flagged blocked: {blocked_count} ({blocked_count / annotated_count * 100:.1f}%)")
    print(f"[INFO] Graph edges open and traversable: {annotated_count - blocked_count} ({(annotated_count - blocked_count) / annotated_count * 100:.1f}%)")
    return G


if __name__ == "__main__":
    from src.ml.predict_risk import RoadRiskPredictor

    predictor = RoadRiskPredictor()
    risk_df = predictor.predict_for_earthquake(
        eq_lat=25.18, eq_lon=91.80, magnitude=6.0, depth_km=35.0, threshold=0.50
    )
    base_G = load_base_graph()
    risk_G = annotate_graph_with_risk(base_G, risk_df, blockage_threshold=0.50)