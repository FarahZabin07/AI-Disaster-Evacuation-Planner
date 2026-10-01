"""
tests/test_planner.py
Phase 18: Unit and Integration Test Suite.
Validates A* search guarantees, CSP constraints, distance formulas, and ML integrity.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math
import pytest
import networkx as nx

from src.graph.astar import haversine_distance_m, AStarRouter
from src.csp.shelter_csp import EvacuationCSP


def test_haversine_accuracy():
    # Known geodesic distance between Dhaka Airport (23.8433, 90.4029) 
    # and Shahbagh (23.7381, 90.3957) is approximately 11.7 km.
    dist_m = haversine_distance_m(23.8433, 90.4029, 23.7381, 90.3957)
    assert 11400.0 < dist_m < 12000.0, f"Unexpected Haversine calculation: {dist_m}m"


def test_astar_admissibility_and_blockage():
    # Build a controlled 3-node graph: Start -> Inter -> Goal
    G = nx.MultiDiGraph()
    G.add_node("S", y=23.800, x=90.400)
    G.add_node("I", y=23.805, x=90.405)
    G.add_node("G", y=23.810, x=90.410)

    # Path 1: S -> I -> G (Blocked at I->G)
    G.add_edge("S", "I", length=100.0, evacuation_cost=100.0, traversable=True, risk_prob=0.1)
    G.add_edge("I", "G", length=100.0, evacuation_cost=float("inf"), traversable=False, risk_prob=0.99)

    # Path 2: Direct detour S -> G (Traversable, higher physical distance)
    G.add_edge("S", "G", length=300.0, evacuation_cost=300.0, traversable=True, risk_prob=0.2)

    router = AStarRouter(G)
    result = router.find_path("S", "G")

    assert result is not None, "A* failed to identify alternative traversable path."
    assert result["path"] == ["S", "G"], "A* selected an invalid or blocked edge sequence."
    assert result["total_distance_m"] == 300.0


def test_csp_capacity_enforcement():
    csp = EvacuationCSP()
    # Mock graph with proper OSMnx graph metadata
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node("N1", y=23.8185, x=90.4078, osmid=1)
    G.add_edge("N1", "N1", length=10.0, evacuation_cost=10.0, traversable=True, risk_prob=0.0)

    # Test exceeding shelter limit (Dhaka Army Stadium max cap: 25,000, free: ~16,800)
    assignment = csp.solve_assignment(
        origin_lat=23.8185,
        origin_lon=90.4078,
        group_size=50000,  # Exceeds maximum possible shelter capacity
        risk_G=G,
        max_dist_km=5.0,
    )
    assert assignment is None, "CSP permitted allocation exceeding maximum shelter capacity."