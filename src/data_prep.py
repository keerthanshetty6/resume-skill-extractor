from datasets import load_dataset
import pandas as pd
import os

#Create folders if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load nathansutton (AI/DS focused - 2,900 rows)
print("Loading nathansutton/data-science-job-descriptions")
ds1 = load_dataset("nathansutton/data-science-job-descriptions", split="train")
df1 = ds1.to_pandas()
df1 = df1[["title", "job_description"]].copy()
df1["source"] = "nathansutton"
print(f"Loaded {len(df1)} rows from nathansutton")
print(df1.head())

# Load batuhanmtl (IT category only)
print("\nLoading batuhanmtl/job-skill-set")
ds2 = load_dataset("batuhanmtl/job-skill-set", split="train")
df2 = ds2.to_pandas()
df2 = df2[df2["category"] == "INFORMATION-TECHNOLOGY"].copy()
df2 = df2.rename(columns={"job_title": "title"})
df2 = df2[["title", "job_description"]].copy()
df2["source"] = "batuhanmtl"
print(f"Loaded {len(df2)} IT rows from batuhanmtl")
print(df2.head())


# Combine
df = pd.concat([df1, df2], ignore_index=True)
print(f"\nTotal rows combined: {len(df)}")

# Basic cleaning 
before = len(df)
df = df.dropna(subset=["title", "job_description"])
df = df[df["job_description"].str.len() >= 200]
df["job_description"] = df["job_description"].str.replace(r'\s+', ' ', regex=True).str.strip() # clean up messy spacing in the text
df = df.drop_duplicates(subset=["job_description"])
df = df.reset_index(drop=True)
after = len(df)
print(f"Rows removed during cleaning: {before - after}")
print(f"Final clean dataset size:     {after}")

# Save
output_path = "data/raw/combined_jobs.json"
df.to_json(output_path, orient="records", lines=True)
print(f"\nSaved to {output_path}")

# Summary
print("\n--- Source breakdown ---")
print(df["source"].value_counts())

print("\n--- Sample titles ---")
print(df["title"].value_counts().head(10))
