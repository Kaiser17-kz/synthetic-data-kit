import pandas as pd
from collections import Counter, defaultdict

#########################################################
# 🔹 LOAD DATASET
#########################################################

df = pd.read_csv("Synthetic Data files/synthetic-data-kit/Healthcare.csv")

print(df.info())
print(df.head())
print(df.isnull().sum())

#########################################################
# 🔹 CLEAN DATA
#########################################################

df["Gender"] = df["Gender"].fillna("Unknown")
df["Symptoms"] = df["Symptoms"].fillna("")

df = df.dropna(subset=["Disease"])  # keep for reference only

df["Disease"] = df["Disease"].str.lower().str.strip()
df["Symptoms"] = df["Symptoms"].str.lower().str.strip()
df["Gender"] = df["Gender"].str.capitalize()

#########################################################
# 🔥 ONE-HOT ENCODING FOR SYMPTOMS
#########################################################

df["Symptoms_List"] = df["Symptoms"].str.split(",")

df["Symptoms_List"] = df["Symptoms_List"].apply(
    lambda x: [s.strip() for s in x if s.strip() != ""]
)

# Get all unique symptoms
all_symptoms = set()
for symptoms in df["Symptoms_List"]:
    all_symptoms.update(symptoms)

all_symptoms = sorted(all_symptoms)

# Create one-hot encoded columns
for symptom in all_symptoms:
    col_name = symptom.replace(" ", "_")
    df[col_name] = df["Symptoms_List"].apply(
        lambda x: 1 if symptom in x else 0
    )

#########################################################
# 🔥 SYMPTOM → BODY PART (More speciality based symptoms)
#########################################################

symptom_to_bodypart = {
    # Head / brain
    "headache": "head",
    "dizziness": "head",
    "migraine": "head",

    # Eyes
    "blurred vision": "eyes",

    # Chest / heart / lungs
    "chest pain": "chest",
    "shortness of breath": "chest",
    "cough": "chest",

    # ENT
    "runny nose": "nose",
    "sneezing": "nose",
    "sore throat": "throat",

    # Digestive
    "abdominal_pain": "abdomen",
    "appetite_loss": "abdomen",
    "nausea": "abdomen",
    "vomiting": "abdomen",
    "diarrhea": "abdomen",

    # Skin
    "rash": "skin",

    # Musculoskeletal
    "back pain": "back",
    "muscle pain": "muscle",
    "joint pain": "joint",

    # General
    "fatigue": "whole body",
    "fever": "whole body",
    "sweating": "whole body",
    "weight loss": "whole body",
    "weight gain": "whole body",
    "swelling": "whole body",

    #Brain
    "anxiety": "brain",
    "depression": "brain",
    "insomnia": "brain",

}

#########################################################
# BODY PART → SPECIALTY
#########################################################

bodypart_to_specialty = {
    "head": "neurology",
    "brain": "neurology",
    "chest": "cardiology",   # default
    "eyes": "ophthalmology",
    "nose": "ent",
    "throat": "ent",
    "abdomen": "gastroenterology",
    "skin": "dermatology",
    "back": "orthopedics",
    "muscle": "orthopedics",
    "joint": "orthopedics",
    "whole body": "general"
}

#########################################################
# SYMPTOM WEIGHTS (1- 4 scale) - higher means more likely to require specialist care
#########################################################

symptom_weight = {
    # Head / Brain (moderate importance)
    "headache": 2,
    "dizziness": 2,
    "migraine": 3,   # more specific than headache

    # Eyes
    "blurred vision": 3,  # often serious, may indicate neurological issues

    # Chest / Heart / Lungs (high importance)
    "chest pain": 4,              # critical symptom
    "shortness of breath": 4,     # critical symptom
    "cough": 2,                   # less specific

    # ENT
    "runny nose": 1,              # common symptom
    "sneezing": 1,                # common symptom
    "sore throat": 2,             # slightly more specific

    # Digestive
    "abdominal_pain": 3,          # moderate-high importance
    "appetite_loss": 2,            # moderate
    "nausea": 2,                  # moderate
    "vomiting": 3,                # high importance
    "diarrhea": 3,                # high importance

    # Skin
    "rash": 2,                     # moderate

    # Musculoskeletal
    "back pain": 2,                # moderate
    "muscle pain": 2,              # moderate
    "joint pain": 2,               # moderate

    # General / Systemic
    "fatigue": 1,                  # low specificity
    "fever": 2,                    # moderate
    "sweating": 1,                 # low
    "weight loss": 2,              # moderate
    "weight gain": 1,              # low
    "swelling": 2,                 # moderate

    # Brain / Mental Health
    "anxiety": 2,                  # moderate
    "depression": 2,               # moderate
    "insomnia": 1                  # lower specificity
}

#########################################################
# 🔥 MAIN ANALYSIS FUNCTION
#########################################################

def analyze_all(row):
    bodyparts_detected = set()
    spec_count = Counter()

    for symptom in all_symptoms:
        col_name = symptom.replace(" ", "_")
        if row[col_name] == 1:
            # Map to body part
            part = symptom_to_bodypart.get(symptom, "whole body")
            bodyparts_detected.add(part)

            # Override specialty rules
            if symptom in ["shortness of breath", "cough"]:
                spec = "pulmonology"
            elif symptom in ["chest pain", "palpitations"]:
                spec = "cardiology"
            elif symptom in ["skin rash", "rash", "itching"]:
                spec = "dermatology"
            else:
                spec = bodypart_to_specialty.get(part, "general practice")

            # Add weighted score
            weight = symptom_weight.get(symptom, 1)
            spec_count[spec] += weight

    # Default case
    if not bodyparts_detected:
        return pd.Series([["whole body"], "general practice", "general practice:1"])

    # Specialty breakdown as string
    spec_breakdown = ", ".join([f"{k}:{v}" for k, v in spec_count.items()])

    return pd.Series([list(bodyparts_detected), spec_count.most_common(1)[0][0], spec_breakdown])

#########################################################
# 🔥 APPLY PIPELINE
#########################################################

df[["Body_Part_Breakdown", "Speciality", "Speciality_Breakdown"]] = df.apply(analyze_all, axis=1)

# Drop Symptoms
df = df.drop(columns=["Symptoms"])

#########################################################
# 🔥 SAVE FINAL DATASET
#########################################################

df.to_csv("cleaned_symptoms_all_bodyparts.csv", index=False)

#########################################################
# 🔥 PREVIEW RESULTS
#########################################################

print("\n✅ FINAL DATASET:")
print(df[["Body_Part_Breakdown", "Speciality", "Speciality_Breakdown"]].head())