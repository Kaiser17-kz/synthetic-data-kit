import pandas as pd

# Load dataset
df = pd.read_csv("Synthetic Data files/synthetic-data-kit/Healthcare.csv")

# Handle missing values
df["Gender"] = df["Gender"].fillna("Unknown")
df["Symptoms"] = df["Symptoms"].fillna("")
df = df.dropna(subset=["Disease"])

# Standardize text
df["Disease"] = df["Disease"].str.lower().str.strip()
df["Symptoms"] = df["Symptoms"].str.lower().str.strip()
df["Gender"] = df["Gender"].str.capitalize()

# Mapping disease → specialty
disease_to_specialty = {
    "common cold": "general",
    "influenza": "general",
    "covid-19": "general",
    "pneumonia": "pulmonology",
    "tuberculosis": "pulmonology",
    "diabetes": "endocrinology",
    "hypertension": "cardiology",
    "asthma": "pulmonology",
    "heart disease": "cardiology",
    "chronic kidney disease": "nephrology",
    "gastritis": "gastroenterology",
    "food poisoning": "general",
    "irritable bowel syndrome (ibs)": "gastroenterology",
    "liver disease": "gastroenterology",
    "ulcer": "gastroenterology",
    "migraine": "neurology",
    "epilepsy": "neurology",
    "stroke": "neurology",
    "dementia": "neurology",
    "parkinson's": "neurology",
    "allergy": "immunology",
    "arthritis": "rheumatology",
    "anemia": "hematology",
    "thyroid disorder": "endocrinology",
    "obesity": "endocrinology",
    "depression": "psychiatry",
    "anxiety": "psychiatry",
    "dermatitis": "dermatology",
    "sinusitis": "ent",
    "bronchitis": "pulmonology"
}
df["Speciality"] = df["Disease"].map(disease_to_specialty).fillna("general")

#########################################################
# 🔹 Body part mapping based on disease
#########################################################

disease_to_affected_bodypart = {
    "common cold": "nose, throat",
    "influenza": "whole body",
    "covid-19": "lungs, whole body",
    "pneumonia": "lungs",
    "tuberculosis": "lungs",
    "diabetes": "pancreas, whole body",
    "hypertension": "heart, blood vessels",
    "asthma": "lungs",
    "heart disease": "heart",
    "chronic kidney disease": "kidneys",
    "gastritis": "stomach",
    "food poisoning": "stomach, intestines",
    "irritable bowel syndrome (ibs)": "intestines",
    "liver disease": "liver",
    "ulcer": "stomach",
    "migraine": "head",
    "epilepsy": "brain",
    "stroke": "brain",
    "dementia": "brain",
    "parkinson's": "brain",
    "allergy": "skin, respiratory system",
    "arthritis": "joints",
    "anemia": "blood",
    "thyroid disorder": "thyroid gland",
    "obesity": "whole body",
    "depression": "brain",
    "anxiety": "brain",
    "dermatitis": "skin",
    "sinusitis": "sinuses",
    "bronchitis": "lungs"
}

# Map disease based on body part
df["Affected_BodyPart"] = df["Disease"].map(disease_to_affected_bodypart).fillna("unspecified")

#########################################################
# 🔹 Map emergency level based on disease
#########################################################

disease_to_emergency = {
    # LOW (pharmacy / mild)
    "common cold": "low",
    "allergy": "low",
    "dermatitis": "low",
    "sinusitis": "low",

    # MEDIUM (clinic / doctor)
    "influenza": "medium",
    "covid-19": "medium",
    "diabetes": "medium",
    "hypertension": "medium",
    "asthma": "medium",
    "gastritis": "medium",
    "food poisoning": "medium",
    "irritable bowel syndrome (ibs)": "medium",
    "liver disease": "medium",
    "ulcer": "medium",
    "migraine": "medium",
    "arthritis": "medium",
    "anemia": "medium",
    "thyroid disorder": "medium",
    "obesity": "medium",
    "depression": "medium",
    "anxiety": "medium",
    "bronchitis": "medium",

    # HIGH (hospital / urgent)
    "pneumonia": "high",
    "tuberculosis": "high",
    "heart disease": "high",
    "chronic kidney disease": "high",
    "epilepsy": "high",
    "stroke": "high",
    "dementia": "high",
    "parkinson's": "high"
}

# Map emergency level based on disease
df["Emergency_Level"] = df["Disease"].map(disease_to_emergency).fillna("medium")

#########################################################
# 🔹 Define facility type based on emergency level
#########################################################
# Map emergency level to recommended facility type
def map_facility_type(emergency, specialty):
    specialty = str(specialty).lower().strip()
    
    if emergency == "high":
        return "hospital"
    elif emergency == "medium":
        # If specialty is general, assign "clinic" for general consultation
        # Otherwise, assign "doctor" for specialist
        if specialty == "general":
            return "clinic"
        else:
            return "doctor"
    elif emergency == "low":
        return "pharmacy"
    
    # Default fallback
    return "clinic"


# Apply mapping
df["Recommended_Facility_Type"] = df.apply(
    lambda row: map_facility_type(row["Emergency_Level"], row["Speciality"]),
    axis=1
)

# Save the cleaned dataset
df.to_csv("cleaned_symptoms_dataset_bodyparts_by_disease.csv", index=False)

print(df.head())