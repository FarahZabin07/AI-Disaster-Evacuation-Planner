"""
src/data/create_shelters.py
Phase 14: Generates the Dhaka Emergency Shelter dataset.
Locations represent major open stadiums, institutional fields, and universities.
Capacities and initial occupancies are simulated for disaster response testing.
"""

import os
import pandas as pd

SHELTER_OUTPUT_PATH = "data/processed/dhaka_shelters.csv"

DHAKA_SHELTERS = [
    {
        "shelter_id": "SH_01",
        "name": "Dhaka Army Stadium, Airport Road",
        "latitude": 23.8185,
        "longitude": 90.4078,
        "capacity": 25000,
        "current_occupancy": 8200,      # Simulated initial intake
        "accessibility": "high",
        "status": "OPEN",
    },
    {
        "shelter_id": "SH_02",
        "name": "Mirpur Sher-e-Bangla National Stadium",
        "latitude": 23.8069,
        "longitude": 90.3634,
        "capacity": 30000,
        "current_occupancy": 12500,     # Simulated initial intake
        "accessibility": "high",
        "status": "OPEN",
    },
    {
        "shelter_id": "SH_03",
        "name": "Uttara Sector 3 Public Field",
        "latitude": 23.8687,
        "longitude": 90.3986,
        "capacity": 8000,
        "current_occupancy": 6800,      # Simulated near-capacity
        "accessibility": "moderate",
        "status": "OPEN",
    },
    {
        "shelter_id": "SH_04",
        "name": "Banani Playground & Park",
        "latitude": 23.7937,
        "longitude": 90.4046,
        "capacity": 10000,
        "current_occupancy": 3100,
        "accessibility": "high",
        "status": "OPEN",
    },
    {
        "shelter_id": "SH_05",
        "name": "BUET Central Field, Palashi",
        "latitude": 23.7266,
        "longitude": 90.3922,
        "capacity": 15000,
        "current_occupancy": 4500,
        "accessibility": "high",
        "status": "OPEN",
    },
    {
        "shelter_id": "SH_06",
        "name": "Dhanmondi Ground (Abahani Field)",
        "latitude": 23.7485,
        "longitude": 90.3753,
        "capacity": 12000,
        "current_occupancy": 11800,     # Simulated almost full
        "accessibility": "moderate",
        "status": "OPEN",
    },
]


def generate_shelter_dataset(output_path: str = SHELTER_OUTPUT_PATH) -> pd.DataFrame:
    df = pd.DataFrame(DHAKA_SHELTERS)
    df["available_capacity"] = df["capacity"] - df["current_occupancy"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dhaka shelter dataset generated at '{output_path}'.")
    return df


if __name__ == "__main__":
    df = generate_shelter_dataset()
    print("\n--- Dhaka Emergency Shelter Registry ---")
    print(df[["shelter_id", "name", "capacity", "available_capacity", "accessibility"]])