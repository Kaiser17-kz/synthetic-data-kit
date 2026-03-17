import pandas as pd
import json

# -----------------------------
# Step 1: Load symptoms dataset
# -----------------------------
symptoms_df = pd.read_csv("medquad.csv")  # Replace with your file path

# -----------------------------
# Step 2: Map focus_area to specialist
# -----------------------------
# Update this mapping with all relevant focus areas
specialist_map = {
    "Glaucoma": "ophthalmology",
    "Diabetes": "endocrinology",
    "Heart Disease": "cardiology",
    "Hypertension": "cardiology",
    "Asthma": "pulmonology",
    "Skin Conditions": "dermatology",
    "Bone Fracture": "orthopaedics",
    "Back Pain": "orthopaedics",
    "Mental Health": "psychiatry",
    "Tooth Pain": "dentist",
    # Add more as needed
}

# Create a specialist column
symptoms_df["specialist"] = symptoms_df["focus_area"].map(specialist_map)

# Drop rows where mapping failed
symptoms_df = symptoms_df.dropna(subset=["specialist"])

# -----------------------------
# Step 3: Prepare Alpaca-style dataset
# -----------------------------
# Rename columns for Alpaca-style prompts
symptoms_df = symptoms_df.rename(columns={
    "question": "instruction",
    "answer": "output"
})
symptoms_df["input"] = ""  # optional extra input

# -----------------------------
# Step 4: Save as JSONL for fine-tuning
# -----------------------------
output_file = "medquad_alpaca.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in symptoms_df.iterrows():
        json_record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"]
        }
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

print(f"Alpaca-style training dataset saved: {output_file}")
print(f"Total records: {len(symptoms_df)}")

# -----------------------------
# Optional: Preview first 2 entries
# -----------------------------
print(symptoms_df[["instruction", "output", "specialist"]].head(2))