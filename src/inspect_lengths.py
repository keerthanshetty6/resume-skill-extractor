import pandas as pd
from datasets import load_dataset

DATASET_ID = "k10shetty/resume-skill-extractor-dataset"

def main():
    print(f"Loading dataset directly from Hugging Face: {DATASET_ID}...\n")
    
    # 1. Load the dataset from the Hugging Face Hub
    dataset = load_dataset(DATASET_ID, split="train")
    
    # 2. Convert to a Pandas DataFrame for super fast sorting and math
    df = dataset.to_pandas()
    
    # 3. Create a new column that stores the character length of the job description
    df['description_length'] = df['job_description'].apply(len)
    #df1 = df[df['description_length']<6500]
    #print(df1.shape)
    # 4. Sort the DataFrame from longest to shortest and grab the top 10
    top_10 = df.sort_values(by='description_length', ascending=False).head(10)
    
    # 5. Display the results nicely
    print("🏆 Top 10 Longest Job Descriptions in the Dataset:")
    print("-" * 75)
    print(f"{'Length (Chars)':<20} | {'Job Title'}")
    print("-" * 75)
    
    for _, row in top_10.iterrows():
        # Print the length and the job title
        print(f"{row['description_length']:<20,d} | {row['title']}")

    # Let's also print the average so you know the baseline!
    avg_length = int(df['description_length'].mean())
    print("-" * 75)
    print(f"\nAverage Job Description Length: {avg_length:,d} characters")
    
    # Quick token estimate (1 token ≈ 4 characters)
    print(f"Average Token Count: ~{avg_length // 4:,d} tokens")

if __name__ == "__main__":
    main()