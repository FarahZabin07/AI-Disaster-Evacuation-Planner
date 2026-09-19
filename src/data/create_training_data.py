import pandas as pd
import numpy as np

# -----------------------------
# 1. Load datasets
# -----------------------------

earthquakes = pd.read_csv("data/raw/earthquake_events.csv")
roads = pd.read_csv("data/processed/roads.csv")

print("Earthquakes:", len(earthquakes))
print("Roads:", len(roads))


# -----------------------------
# 2. Haversine distance
# -----------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# -----------------------------
# 3. Create earthquake-road pairs
# -----------------------------

records = []

for _, earthquake in earthquakes.iterrows():

    for _, road in roads.iterrows():

        distance = haversine(
            earthquake["latitude"],
            earthquake["longitude"],
            (road["start_lat"] + road["end_lat"]) / 2,
            (road["start_lon"] + road["end_lon"]) / 2
        )

        # Simple prototype damage/blockage proxy
        risk_score = earthquake["mag"] / (distance + 1)

        blocked_proxy = 1 if risk_score >= 0.25 else 0

        records.append({
            "earthquake_id": earthquake["id"],
            "road_id": road["road_id"],
            "magnitude": earthquake["mag"],
            "earthquake_depth_km": earthquake["depth"],
            "distance_from_epicenter_km": distance,
            "road_type": road["road_type"],
            "oneway": road["oneway"],
            "road_length_m": road["road_length_m"],
            "lanes": road["lanes"],
            "surface": road["surface"],
            "maxspeed": road["maxspeed"],
            "blocked_proxy": blocked_proxy
        })


# -----------------------------
# 4. Create DataFrame
# -----------------------------

training_data = pd.DataFrame(records)


# -----------------------------
# 5. Save
# -----------------------------

output_file = "data/processed/training_data.csv"

training_data.to_csv(output_file, index=False)

print("\nTraining dataset created successfully!")
print("Saved to:", output_file)
print("Total rows:", len(training_data))

print("\nBlocked proxy distribution:")
print(training_data["blocked_proxy"].value_counts())

print("\nFirst 5 rows:")
print(training_data.head())