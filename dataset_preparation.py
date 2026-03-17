import pandas as pd

# Load CSV
df = pd.read_csv("hotosm_mys_health_facilities_points_geojson.csv")

# Rename columns for clarity
df = df.rename(columns={
    "healthcare:speciality": "speciality",
    "capacity:persons": "capacity",
    "X": "longitude",
    "Y": "latitude"
})

# Select important columns that exist
columns_needed = [
    "name",
    "amenity",
    "healthcare",
    "speciality",
    "addr:full",
    "addr:city",
    "latitude",
    "longitude",
    "capacity"
]
columns_needed = [col for col in columns_needed if col in df.columns]
df = df[columns_needed]

# Remove rows without facility name
df = df.dropna(subset=["name"])

# Fill missing values
if "speciality" in df.columns:
    df["speciality"] = df["speciality"].fillna("general")
if "addr:full" in df.columns:
    df["addr:full"] = df["addr:full"].fillna("Unknown")
if "addr:city" in df.columns:
    df["addr:city"] = df["addr:city"].fillna("Unknown")
if "capacity" in df.columns:
    df["capacity"] = pd.to_numeric(df["capacity"], errors='coerce').fillna(0)

# Remove duplicates
df = df.drop_duplicates(subset=["name", "latitude", "longitude"])

# Ensure speciality column is consistent
# Lowercase everything to avoid mismatches
df["speciality"] = df["speciality"].str.lower()
df["specialist"] = df["specialist"].str.lower()

# Map all amenities to 3 categories only
def map_amenity(x):
    if x in ["hospital", "clinic", "pharmacy"]:
        return x
    else:
        return "other"  # everything else goes to "other"

df["amenity"] = df["amenity"].apply(map_amenity)

# Save cleaned dataset
df.to_csv("clean_healthcarefacilities_final.csv", index=False)

# Check results
print("Amenity value counts after mapping:")
print(df["amenity"].value_counts())