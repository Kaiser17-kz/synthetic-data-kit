import pandas as pd

# Load dataset
df = pd.read_csv("Synthetic Data files/synthetic-data-kit/Healthcare.csv")

# Inspect data
print(df.info())
print(df.head())
print(df.isnull().sum())

# Handle missing values
df["Gender"] = df["Gender"].fillna("Unknown")
df["Symptoms"] = df["Symptoms"].fillna("")

# Drop rows with missing target
df = df.dropna(subset=["Disease"])

# Standardize text
df["Disease"] = df["Disease"].str.lower().str.strip()
df["Symptoms"] = df["Symptoms"].str.lower().str.strip()
df["Gender"] = df["Gender"].str.capitalize()

#########################################################
# 🔹 SPLIT SYMPTOMS INTO MULTIPLE COLUMNS
#########################################################

# Split symptoms by comma
symptoms_split = df["Symptoms"].str.split(",", expand=True)

# Clean each column (remove extra spaces)
symptoms_split = symptoms_split.apply(lambda col: col.str.strip())

# Rename columns dynamically
symptoms_split.columns = [f"Symptom_{i+1}" for i in range(symptoms_split.shape[1])]

# Merge back into main dataframe
df = pd.concat([df, symptoms_split], axis=1)

#########################################################
# Mapping disease → specialist
#########################################################

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
    "parkinson’s disease": "neurology",

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

# Save cleaned dataset
df.to_csv("cleaned_symptoms_dataset_noencode.csv", index=False)

print(df.head())