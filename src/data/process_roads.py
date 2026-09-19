import geopandas as gpd
import pandas as pd

# Read raw road data
input_file = "data/raw/uttara_roads.geojson"
roads = gpd.read_file(input_file)

print("Total roads:", len(roads))
print("Available columns:")
print(roads.columns.tolist())

# Convert to metric CRS for accurate length calculation
roads_metric = roads.to_crs(epsg=32646)

# Create processed dataset
data = pd.DataFrame()

data["road_id"] = roads["@id"].astype(str)
data["road_type"] = roads["highway"].fillna("unknown")
data["oneway"] = roads["oneway"].fillna("no")

# Optional OpenStreetMap information
data["road_name"] = roads["name"] if "name" in roads.columns else ""
data["lanes"] = roads["lanes"] if "lanes" in roads.columns else ""
data["surface"] = roads["surface"] if "surface" in roads.columns else ""
data["maxspeed"] = roads["maxspeed"] if "maxspeed" in roads.columns else ""

# Calculate road length
data["road_length_m"] = roads_metric.geometry.length

# Starting coordinates
data["start_lon"] = roads.geometry.apply(lambda x: x.coords[0][0])
data["start_lat"] = roads.geometry.apply(lambda x: x.coords[0][1])

# Ending coordinates
data["end_lon"] = roads.geometry.apply(lambda x: x.coords[-1][0])
data["end_lat"] = roads.geometry.apply(lambda x: x.coords[-1][1])

# Save processed data
output_file = "data/processed/roads.csv"
data.to_csv(output_file, index=False)

print("\nRoad dataset created successfully!")
print("Saved to:", output_file)
print("Number of roads:", len(data))

print("\nFirst 5 rows:")
print(data.head())