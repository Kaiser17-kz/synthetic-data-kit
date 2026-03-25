import pandas as pd

#########################################################
##//check for unique values in amenity and healthcare//##
df = pd.read_csv("Synthetic Data files\synthetic-data-kit\hotosm_mys_health_facilities_points_geojson.csv")

print(df["amenity"].unique())
print(df["healthcare"].unique())
print(df["healthcare:speciality"].unique())
print(df["amenity"].value_counts())

#Normalize text
df["amenity"] = df["amenity"].str.lower().str.strip()
df["healthcare"] = df["healthcare"].str.lower().str.strip()

##//Categorize into 3 different fields//##
#map and categorize to 3 related fields of hospital, pharmacy and clinic

mapping = {
    # Hospital-related
    "hospital": "hospital",
    "ward": "hospital",
    "maternity": "hospital",
    "birthing_centre": "hospital",
    "hospice": "hospital",

    # Pharmacy-related
    "pharmacy": "pharmacy",
    "drugstore": "pharmacy",

    # Clinic-related (includes dentist + others)
    "clinic": "clinic",
    "doctors": "clinic",
    "doctor": "clinic",
    "dentist": "clinic",
    "physiotherapist": "clinic",
    "psychotherapist": "clinic",
    "optometrist": "clinic",
    "podiatrist": "clinic",
    "midwife": "clinic",
    "rehabilitation": "clinic",
    "dialysis": "clinic",
    "haemodialysis": "clinic",
    "vaccination_centre": "clinic",
    "laboratory": "clinic",
    "blood_donation": "clinic",
    "doctor,_clinic": "clinic"
}

df["amenity_cleaned"] = df["amenity"].map(mapping)

# Create cleaned category column using amenity first
df["amenity_cleaned"] = df["amenity"].map(mapping)

# Fill missing using healthcare column
df["amenity_cleaned"] = df["amenity_cleaned"].fillna(
    df["healthcare"].map(mapping)
)

#########################################################
#//Data cleaning//##

# Rename columns for clarity
df = df.rename(columns={
    "healthcare:speciality": "speciality",
    "X": "longitude",
    "Y": "latitude"
})

# Check duplicates only based on 'name', 'latitude', 'longitude'
print(df.duplicated(subset=["name", "latitude", "longitude"]))

# Remove rows without facility name
df = df.dropna(subset=["name"])

# Drop columns
df = df.drop(columns=["operator:type", "osm_id", "osm_type", "capacity:persons", "name:ms", "amenity", "building", "source", "name:en"])

# Keep only the 3 categories
df_clean = df[df["amenity_cleaned"].isin(["hospital", "pharmacy", "clinic"])]

# Drop rows that still couldn't be classified
df_clean = df_clean.dropna(subset=["amenity_cleaned"])

# Check clean data
print(df_clean["amenity_cleaned"].value_counts())

#########################################################
##//Based on the amenity, the specialty is created based on the amenity which is general, dentist and medication//##

# Check speciality of different facilities
print(df_clean["speciality"].value_counts())

# Save cleaned dataset
df_clean.to_csv("clean_health_facilities.csv", index=False)

#########################################################
##//FILTER ONLY VALID CATEGORIES

df_clean = df[df["amenity_cleaned"].isin(["hospital", "clinic", "pharmacy"])]

# Drop rows that still couldn't be classified
df_clean = df_clean.dropna(subset=["amenity_cleaned"])

#########################################################
##//CREATE NEW SPECIALTY COLUMN


specialty_mapping = {
    "hospital": "general",
    "clinic": "dentist",
    "pharmacy": "medication"
}

df_clean["specialty_cleaned"] = df_clean["amenity_cleaned"].map(specialty_mapping)

#########################################################
##//SAVE CLEAN DATASET

df_clean.to_csv("clean_health_facilities.csv", index=False)

print("\n✅ Data cleaning complete. File saved as 'clean_health_facilities.csv'")



