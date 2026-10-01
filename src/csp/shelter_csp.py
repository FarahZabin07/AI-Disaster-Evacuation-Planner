"""
src/csp/shelter_csp.py
Phase 15: Constraint Satisfaction Problem (CSP) solver for shelter allocation.
Finds feasible and optimal shelters satisfying capacity, accessibility,
traversability, and distance constraints.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import networkx as nx
import osmnx as ox
import pandas as pd

from src.graph.astar import AStarRouter, haversine_distance_m

SHELTER_PATH = "data/processed/dhaka_shelters.csv"


class EvacuationCSP:
    def __init__(self, shelters_path: str = SHELTER_PATH):
        self.shelters_df = pd.read_csv(shelters_path)

    def solve_assignment(
        self,
        origin_lat: float,
        origin_lon: float,
        group_size: int,
        risk_G: nx.MultiDiGraph,
        max_dist_km: float = 18.0,
    ) -> Optional[Dict[str, any]]:
        """
        CSP Solver:
        Variables: Evacuee Group (group_size, origin)
        Domain: Candidate Shelters
        Constraints:
          - Capacity: available_capacity >= group_size
          - Status: status == 'OPEN'
          - Distance: straight line <= max_dist_km
          - Routing: A* path must exist in risk-annotated graph
        Objective: Minimize A* risk-augmented evacuation cost.
        """
        origin_node = ox.distance.nearest_nodes(risk_G, X=origin_lon, Y=origin_lat)
        router = AStarRouter(risk_G)

        feasible_solutions = []

        print(f"\n[INFO] Evaluating CSP constraints for {group_size} evacuees from ({origin_lat:.4f}, {origin_lon:.4f})...")

        for _, shelter in self.shelters_df.iterrows():
            s_id = shelter["shelter_id"]
            s_name = shelter["name"]

            # Constraint 1: Status Check
            if shelter["status"] != "OPEN":
                print(f"  [REJECT] {s_name}: Shelter is CLOSED.")
                continue

            # Constraint 2: Capacity Check
            if shelter["available_capacity"] < group_size:
                print(f"  [REJECT] {s_name}: Insufficient capacity ({shelter['available_capacity']} available vs {group_size} needed).")
                continue

            # Constraint 3: Geodesic Maximum Distance Check
            straight_dist_m = haversine_distance_m(
                origin_lat, origin_lon, shelter["latitude"], shelter["longitude"]
            )
            straight_dist_km = straight_dist_m / 1000.0
            if straight_dist_km > max_dist_km:
                print(f"  [REJECT] {s_name}: Exceeds maximum distance ({straight_dist_km:.1f} km > {max_dist_km} km).")
                continue

            # Constraint 4: Network Traversability (A* Path Check)
            shelter_node = ox.distance.nearest_nodes(
                risk_G, X=shelter["longitude"], Y=shelter["latitude"]
            )
            route_result = router.find_path(origin_node, shelter_node)

            if not route_result:
                print(f"  [REJECT] {s_name}: No traversable route (blocked by earthquake damage).")
                continue

            print(f"  [FEASIBLE] {s_name}: Path found! Dist={route_result['total_distance_m']/1000.0:.2f}km, Cost={route_result['total_cost']:.1f}")

            feasible_solutions.append(
                {
                    "shelter_id": s_id,
                    "shelter_name": s_name,
                    "shelter_lat": shelter["latitude"],
                    "shelter_lon": shelter["longitude"],
                    "capacity_remaining": shelter["available_capacity"] - group_size,
                    "route": route_result["path"],
                    "route_distance_km": round(route_result["total_distance_m"] / 1000.0, 2),
                    "route_cost": route_result["total_cost"],
                    "avg_risk": route_result["avg_risk"],
                    "explored_nodes": route_result["nodes_explored"],
                }
            )

        if not feasible_solutions:
            print("[WARNING] CSP failed to satisfy all constraints. No feasible shelter available.")
            return None

        # Select optimal shelter by lowest risk-augmented routing cost
        best_assignment = min(feasible_solutions, key=lambda x: x["route_cost"])
        print(f"\n[OPTIMAL ASSIGNMENT] Assigned to: {best_assignment['shelter_name']}")
        return best_assignment


if __name__ == "__main__":
    from src.graph.build_graph import load_base_graph, annotate_graph_with_risk
    from src.ml.predict_risk import RoadRiskPredictor

    predictor = RoadRiskPredictor()
    risk_df = predictor.predict_for_earthquake(25.18, 91.80, 6.0, 35.0, threshold=0.80)
    base_G = load_base_graph()
    risk_G = annotate_graph_with_risk(base_G, risk_df, blockage_threshold=0.80)

    # Test query: 1,500 evacuees starting from Gulshan-2
    csp = EvacuationCSP()
    result = csp.solve_assignment(
        origin_lat=23.7925,
        origin_lon=90.4078,
        group_size=1500,
        risk_G=risk_G,
        max_dist_km=15.0,
    )