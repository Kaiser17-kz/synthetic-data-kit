import pandas as pd

# Load symptoms Q&A dataset
symptoms_df = pd.read_csv("medquad.csv")  # Your medical Q&A file

# Load healthcare facilities dataset
facilities_df = pd.read_csv("clean_healthcarefacilities_final.csv")  # From previous cleaning

# Optional: Map focus_area to specialist
specialist_map = {
    "Glaucoma": "ophthalmology",
    "High Blood Pressure": "cardiology",
    "Diabetes": "endocrinology",
    "Heart Disease": "cardiology",
    # Add more mappings as needed
}
symptoms_df["specialist"] = symptoms_df["focus_area"].map(specialist_map)
symptoms_df = symptoms_df.dropna(subset=["specialist"])

# Map healthcare facility 'speciality' or 'amenity' to specialist
facility_specialist_map = {
    "hospital": ["ophthalmology", "cardiology", "endocrinology", "orthopaedics"],
    "clinic": ["ophthalmology", "cardiology", "endocrinology", "dermatology", "psychiatry"],
    "pharmacy": []  # pharmacies are general
}

# Add specialist column to facilities
def match_specialist(row):
    spec_list = facility_specialist_map.get(row['amenity'], [])
    return ", ".join(spec_list) if spec_list else "general"

facilities_df["specialist"] = facilities_df.apply(match_specialist, axis=1)

##merge 

import random

def generate_instruction(row):
    # Find all facilities matching the specialist
    specialist = row["specialist"]
    matched_facilities = facilities_df[facilities_df["specialist"].str.contains(specialist, case=False)]

    # Pick up to 3 random facilities to include in instruction
    sample_facilities = matched_facilities.sample(n=min(3, len(matched_facilities))) if len(matched_facilities) > 0 else pd.DataFrame()
    facility_list = "\n".join(sample_facilities["name"] + " - " + sample_facilities["addr:full"]) if not sample_facilities.empty else "No nearby facilities available."

    # Build Alpaca-style instruction
    return f"{row['question']} Provide the answer and suggest nearby facilities for specialist '{specialist}'.\nNearby facilities:\n{facility_list}"

# Create new instruction column
symptoms_df["instruction"] = symptoms_df.apply(generate_instruction, axis=1)
symptoms_df["output"] = symptoms_df["answer"]
symptoms_df["input"] = ""

import json

output_file = "medical_alpaca_with_facilities.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in symptoms_df.iterrows():
        json_record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"]
        }
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

print(f"Training dataset saved: {output_file}")
print(f"Total records: {len(symptoms_df)}")