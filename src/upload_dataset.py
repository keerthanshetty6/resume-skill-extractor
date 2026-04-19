import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Dataset object
dataset = load_dataset("json", data_files="data/processed/labeled_jobs_clean.json", split="train")

# train test split
#dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)

# 3. Push to HF
dataset.push_to_hub(
    "keerthanshetty/resume-skill-extractor-dataset",
    token=os.environ["HF_TOKEN"]
)

print("Done.")