import pandas as pd

df = pd.read_json("data/processed/labeled_jobs.json", lines=True)

# Rows to REMOVE (either empty skills or too many skills)
removed = df[(df["required_skills"].apply(len) == 0) | (df["required_skills"].apply(len) > 30)]
print(f"Rows being removed: {len(removed)}")
removed.to_csv("data/processed/removed_rows.csv", index=False)

# Clean dataset
df_clean = df[(df["required_skills"].apply(len) > 0) & (df["required_skills"].apply(len) <= 30)]
df_clean = df_clean.reset_index(drop=True)
print(f"Clean rows remaining: {len(df_clean)}")
df_clean.to_json("data/processed/labeled_jobs_clean.json", orient="records", lines=True)