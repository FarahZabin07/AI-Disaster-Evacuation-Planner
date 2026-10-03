"""
app/app.py
Phase 17: Interactive Streamlit Web Application using st_folium and OpenStreetMap.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import folium
import requests
from streamlit_js_eval import streamlit_js_eval
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

from src.ml.predict_risk import RoadRiskPredictor
from src.graph.build_graph import load_base_graph, annotate_graph_with_risk
from src.csp.shelter_csp import EvacuationCSP

st.set_page_config(
    page_title="AI Disaster Evacuation Planner - Dhaka",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st_autorefresh(interval=300000, key="earthquake_refresh")

st.title("🚨 AI-Based Earthquake Disaster Evacuation Planner (Dhaka)")
st.caption(
    "Integrated system combining USGS Seismic Data, OSM Dhaka Road Graph, "
    "Logistic Regression Hazard Inference, CSP Shelter Allocation, and Handcrafted A* Search."
)

@st.cache_resource(show_spinner="Loading road network graph...")
def get_graph():
    return load_base_graph()

@st.cache_resource(show_spinner="Loading trained ML pipeline...")
def get_predictor():
    return RoadRiskPredictor()

@st.cache_data(ttl=300, show_spinner="Fetching latest earthquakes from USGS...")
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        earthquakes = []

        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            coordinates = feature.get("geometry", {}).get("coordinates", [])

            if len(coordinates) < 3:
                continue

            earthquakes.append({
                "place": properties.get("place", "Unknown"),
                "magnitude": properties.get("mag"),
                "time": pd.to_datetime(properties.get("time"), unit="ms"),
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "depth_km": coordinates[2],
            })

        return pd.DataFrame(earthquakes)

    except Exception as e:
        st.warning(
            f"Could not fetch live USGS data. Using saved earthquake data instead. Error: {e}"
        )
        return pd.read_csv("data/raw/earthquakes.csv")

try:
    base_G = get_graph()
    predictor = get_predictor()
    quakes_df = get_earthquakes()
    csp_solver = EvacuationCSP()
except Exception as e:
    st.error(f"Initialization error: {e}. Please ensure Phases 1 through 15 are completed.")
    st.stop()
# ----------------- GET LIVE LOCATION -----------------
location = streamlit_js_eval(
    js_expressions="""
    new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            (position) => resolve({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracy: position.coords.accuracy
            }),
            (error) => resolve({
                error: error.message
            })
        );
    })
    """,
    key="get_location"
)
# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("1. Seismic Event Setup")
quake_mode = st.sidebar.radio("Earthquake Input Mode", ["Select USGS Event", "Custom Simulation"])

if quake_mode == "Select USGS Event":
    sorted_quakes = quakes_df.sort_values("magnitude", ascending=False).reset_index(drop=True)
    options = [
        f"M{r.magnitude:.1f} | {r.place} ({str(r.time)[:10]})"
        for _, r in sorted_quakes.head(60).iterrows()
    ]
    selected_idx = st.sidebar.selectbox("Choose Historical Event", range(len(options)), format_func=lambda x: options[x])
    eq_row = sorted_quakes.iloc[selected_idx]
    eq_mag = float(eq_row["magnitude"])
    eq_lat = float(eq_row["latitude"])
    eq_lon = float(eq_row["longitude"])
    eq_depth = float(eq_row["depth_km"])
    st.sidebar.info(f"Depth: {eq_depth:.1f} km | Lat: {eq_lat:.2f}, Lon: {eq_lon:.2f}")
else:
    eq_mag = st.sidebar.slider("Magnitude (Mw)", 4.5, 8.5, 6.0, 0.1)
    eq_depth = st.sidebar.slider("Depth (km)", 5.0, 150.0, 25.0, 5.0)
    eq_lat = st.sidebar.number_input("Epicenter Latitude", 20.0, 28.0, 24.30, 0.05)
    eq_lon = st.sidebar.number_input("Epicenter Longitude", 88.0, 93.0, 91.20, 0.05)

st.sidebar.header("2. Evacuation Parameters")
dhaka_neighborhoods = {
    "Gulshan-2 Circle": (23.7925, 90.4078),
    "Banani Road 11": (23.7910, 90.4080),
    "Uttara Sector 3": (23.8687, 90.3986),
    "Mirpur-10 Circle": (23.8069, 90.3687),
    "Dhanmondi 27": (23.7533, 90.3769),
    "Shahbagh / Dhaka Univ": (23.7381, 90.3957),
}
use_live_location = st.sidebar.checkbox(
    "📍 Use My Live Location",
    value=False,
    key="live_location_checkbox"
)

if use_live_location and isinstance(location, dict) and "latitude" in location:
    origin_name = "📍 My Current Location"
    origin_coords = (
        float(location["latitude"]),
        float(location["longitude"])
    )

    st.sidebar.success(
        f"Location detected: {origin_coords[0]:.5f}, {origin_coords[1]:.5f}"
    )

if use_live_location and isinstance(location, dict) and "latitude" in location:
    origin_name = "📍 My Current Location"
    origin_coords = (
        float(location["latitude"]),
        float(location["longitude"])
    )

    st.sidebar.success(
        f"Location detected: {origin_coords[0]:.5f}, {origin_coords[1]:.5f}"
    )

elif use_live_location and isinstance(location, dict) and "error" in location:
    origin_name = st.sidebar.selectbox(
        "Evacuation Origin",
        list(dhaka_neighborhoods.keys())
    )

    origin_coords = dhaka_neighborhoods[origin_name]

    st.sidebar.warning(
        f"Could not get location: {location['error']}"
    )

else:
    origin_name = st.sidebar.selectbox(
        "Evacuation Origin",
        list(dhaka_neighborhoods.keys())
    )

    origin_coords = dhaka_neighborhoods[origin_name]


group_size = st.sidebar.number_input("Evacuee Group Size", min_value=10, max_value=15000, value=800, step=50)
risk_threshold = st.sidebar.slider("Road Severance Threshold P(risk)", 0.50, 0.95, 0.75, 0.05)

# ----------------- PIPELINE EXECUTION -----------------
risk_df = predictor.predict_for_earthquake(
    eq_lat=eq_lat,
    eq_lon=eq_lon,
    magnitude=eq_mag,
    depth_km=eq_depth,
    threshold=risk_threshold,
)

annotated_G = annotate_graph_with_risk(
    base_G, risk_df, blockage_threshold=risk_threshold
)

total_edges = len(risk_df)
blocked_edges = int(risk_df["is_blocked"].sum())
passable_edges = total_edges - blocked_edges

col1, col2, col3, col4 = st.columns(4)
col1.metric("Earthquake Magnitude", f"Mw {eq_mag:.1f}")
col2.metric("Total Roads Evaluated", f"{total_edges:,}")
col3.metric("Severely Blocked Roads", f"{blocked_edges:,}", f"{blocked_edges/total_edges*100:.1f}% cut", delta_color="inverse")
col4.metric("Safe / Passable Roads", f"{passable_edges:,}", f"{passable_edges/total_edges*100:.1f}% open")

assignment = csp_solver.solve_assignment(
    origin_lat=origin_coords[0],
    origin_lon=origin_coords[1],
    group_size=group_size,
    risk_G=annotated_G,
    max_dist_km=25.0,
)

col_map, col_details = st.columns([2.2, 1.0])

with col_details:
    st.subheader("📋 Evacuation Action Plan")
    if assignment:
        st.success(f"**Assigned Shelter:**\n{assignment['shelter_name']}")
        st.write(f"**Route Travel Distance:** `{assignment['route_distance_km']} km`")
        st.write(f"**Average Route Risk:** `{assignment['avg_risk']*100:.1f}%`")
        st.write(f"**Remaining Shelter Capacity:** `{assignment['capacity_remaining']:,} evacuees`")
        st.write(f"**A* Explored Nodes:** `{assignment['explored_nodes']}`")
        st.info("Route computed using risk-penalized edge weights avoiding blocked corridors.")
    else:
        st.error("No feasible shelter found. Try lowering group size or adjusting the threshold.")

    st.markdown("---")
    st.subheader("🎨 Map Legend")
    st.markdown(
        """
        * 🔵 **Blue House Icon:** Your Evacuation Starting Point
        * 🟢 **Green Shield:** Recommended Shelter (Safe & has space)
        * 🟣 **Purple Shield:** Alternative Shelter (Open)
        * 🟠 **Orange Shield:** Shelter Full / Insufficient Capacity
        * 🟦 **Thick Blue Line:** Safe Evacuation Path (A* algorithm)
        * 🟥 **Red Lines:** Blocked Roads (Debris / Structural Failure)
        """
    )

with col_map:
    st.subheader("🗺 Risk-Aware Dhaka Evacuation Map")

    m = folium.Map(
        location=[origin_coords[0], origin_coords[1]],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # 1. Draw Blocked Roads as Red Lines directly on the map
    if blocked_edges > 0:
        blocked_df = risk_df[risk_df["is_blocked"] == 1].head(250)
        for _, row in blocked_df.iterrows():
            if "geometry" in row and row["geometry"] is not None:
                coords = [(pt[1], pt[0]) for pt in row["geometry"].coords]
                folium.PolyLine(
                    coords,
                    color="#FF0000",
                    weight=3,
                    opacity=0.75,
                    tooltip=f"BLOCKED: Road failure risk {row['risk_prob']*100:.1f}%",
                ).add_to(m)

    # 2. Origin Marker
    folium.Marker(
        location=[origin_coords[0], origin_coords[1]],
        popup=f"Origin: {origin_name} ({group_size} evacuees)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # 3. Shelter Markers
    for _, sh in csp_solver.shelters_df.iterrows():
        is_chosen = assignment and (sh["shelter_id"] == assignment["shelter_id"])
        color = "green" if is_chosen else ("orange" if sh["available_capacity"] < group_size else "purple")
        folium.Marker(
            location=[sh["latitude"], sh["longitude"]],
            popup=f"<b>{sh['name']}</b><br>Available: {sh['available_capacity']:,}",
            tooltip=f"{sh['name']} ({sh['available_capacity']} slots)",
            icon=folium.Icon(color=color, icon="shield", prefix="fa"),
        ).add_to(m)

    # 4. Safe Recommended Route (Blue Line)
    if assignment and assignment.get("route"):
        route_nodes = assignment["route"]
        route_coords = [
            (annotated_G.nodes[n]["y"], annotated_G.nodes[n]["x"]) for n in route_nodes
        ]
        folium.PolyLine(
            route_coords,
            color="#0055FF",
            weight=6,
            opacity=0.9,
            tooltip=f"Recommended Evacuation Route ({assignment['route_distance_km']} km)",
        ).add_to(m)

    st_folium(m, width=820, height=560, returned_objects=[])
