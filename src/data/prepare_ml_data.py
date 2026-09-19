import pandas as pd

# Load training data
df = pd.read_csv("data/processed/training_data.csv")

# Keep only useful features
df = df[
    [
        "magnitude",
        "earthquake_depth_km",
        "distance_from_epicenter_km",
        "road_type",
        "oneway",
        "road_length_m",
        "blocked_proxy"
    ]
].copy()

# Convert categorical columns to numbers
df = pd.get_dummies(
    df,
    columns=["road_type", "oneway"],
    drop_first=True
)

# Convert True/False to 0/1
for column in df.columns:
    if df[column].dtype == "bool":
        df[column] = df[column].astype(int)

# Save ML-ready dataset
output_file = "data/processed/ml_ready_data.csv"
df.to_csv(output_file, index=False)

print("ML-ready dataset created successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["blocked_proxy"].value_counts())

print("\nFirst 5 rows:")
print(df.head())
