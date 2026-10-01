"""
src/data/fetch_roads.py
Retrieves and simplifies Dhaka's primary arterial road network from OpenStreetMap.
"""

import os
import osmnx as ox

# Bounding box for Dhaka metropolitan arterial network:
# Covers Old Dhaka through Uttara, Mirpur to Badda
BBOX = {
    "north": 23.8900,
    "south": 23.6800,
    "east": 90.4500,
    "west": 90.3300,
}

OUTPUT_GRAPH = "data/raw/dhaka_network.graphml"
OUTPUT_EDGES = "data/raw/dhaka_edges.geojson"
OUTPUT_NODES = "data/raw/dhaka_nodes.geojson"


def fetch_dhaka_roads(
    bbox: dict = BBOX,
    graph_out: str = OUTPUT_GRAPH,
    edges_out: str = OUTPUT_EDGES,
    nodes_out: str = OUTPUT_NODES,
):
    """Downloads OSM drive network within bbox, saves GraphML and GeoJSON."""
    print(f"[INFO] Fetching drivable street network for Dhaka BBox: {bbox}...")

    # Filter for evacuation-capable vehicular corridors
    cf = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|'
        'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
    )

    try:
        # OSMnx 2.x interface: (left, bottom, right, top)
        G = ox.graph_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type="drive",
            custom_filter=cf,
            simplify=True,
        )
    except TypeError:
        # OSMnx 1.x interface: (north, south, east, west)
        G = ox.graph_from_bbox(
            bbox["north"],
            bbox["south"],
            bbox["east"],
            bbox["west"],
            network_type="drive",
            custom_filter=cf,
            simplify=True,
        )

    print(
        f"[INFO] Raw graph retrieved: {len(G.nodes)} intersections, "
        f"{len(G.edges)} road segments."
    )

    # Convert graph to GeoDataFrames
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # Convert list attributes to strings for GeoJSON compatibility
    for col in gdf_edges.columns:
        if gdf_edges[col].apply(lambda x: isinstance(x, list)).any():
            gdf_edges[col] = gdf_edges[col].astype(str)

    for col in gdf_nodes.columns:
        if gdf_nodes[col].apply(lambda x: isinstance(x, list)).any():
            gdf_nodes[col] = gdf_nodes[col].astype(str)

    os.makedirs(os.path.dirname(graph_out), exist_ok=True)

    # Save GraphML for NetworkX routing
    ox.save_graphml(G, filepath=graph_out)
    print(f"[SUCCESS] Saved graph topology to '{graph_out}'.")

    # Save spatial GeoJSON layers
    gdf_edges.to_file(edges_out, driver="GeoJSON")
    gdf_nodes.to_file(nodes_out, driver="GeoJSON")
    print(f"[SUCCESS] Exported {len(gdf_edges)} edges to '{edges_out}'.")
    print(f"[SUCCESS] Exported {len(gdf_nodes)} nodes to '{nodes_out}'.")

    return G, gdf_nodes, gdf_edges


if __name__ == "__main__":
    G, nodes, edges = fetch_dhaka_roads()
    print("\n--- Network Attributes Sample ---")
    print(edges[["highway", "length", "oneway"]].head(5))