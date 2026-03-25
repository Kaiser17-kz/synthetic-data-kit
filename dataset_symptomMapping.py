import pandas as pd
import random

#########################################################
## LOAD CLEANED SYMPTOMS DATASET
#########################################################

df = pd.read_csv(r"cleaned_symptoms_dataset.csv")

#########################################################
## CLEAN DATA
#########################################################

df["Symptoms"] = df["Symptoms"].str.lower().str.strip()
df["Disease"] = df["Disease"].str.lower().str.strip()
df["Speciality"] = df["Speciality"].str.lower().str.strip()

#########################################################
## INSTRUCTIONS (better for LLM training)
#########################################################

instructions_list = [
    "Identify the most likely disease and recommend the appropriate medical specialist.",
    "Based on the symptoms, determine the disease and suggest the correct specialist.",
    "Analyze the symptoms and provide the diagnosis along with the appropriate specialist.",
    "Given the symptoms, what disease is likely and which specialist should be consulted?"
]

#########################################################
## GENERATE ALPACA DATASET
#########################################################

target_rows = 1000
data = []

# Repeat until we reach 2000 rows
while len(data) < target_rows:
    row = df.sample(1).iloc[0]  # sample with replacement
    
    symptoms = row["Symptoms"]
    disease = row["Disease"]
    specialist = row["Speciality"]
    
    # Random instruction
    instruction = random.choice(instructions_list)
    
    # Input
    input_text = f"Symptoms: {symptoms}"
    
    # Output
    output_text = (
        f"Disease: {disease}\n"
        f"Speciality: {specialist}"
    )
    
    data.append({
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output_text.strip()
    })

#########################################################
## SAVE DATASET
#########################################################

alpaca_df = pd.DataFrame(data)

alpaca_df.to_csv("alpaca_symptoms_dataset_2000.csv", index=False)

print(alpaca_df.head())
print(f"\n✅ Alpaca-format dataset with {len(alpaca_df)} rows ready for training!")