from google import genai
from google.genai import types
import pandas as pd
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run on 5 rows only")
args = parser.parse_args()

# Load variables from the .env file into the script
load_dotenv()

# Config 
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
INPUT_PATH     = "data/raw/combined_jobs.json"
OUTPUT_PATH    = "data/processed/labeled_jobs.json"
#REQUESTS_PER_MINUTE = 14        
DELAY = 0.5 #60 / REQUESTS_PER_MINUTE

client = genai.Client(api_key=GEMINI_API_KEY)


# Prompt template
def build_prompt(job_description):
    return f"""You are an expert technical recruiter. Read the job description below.
Extract and return a JSON object with exactly these two fields:
- summary: a 2 sentence summary of the role
- required_skills: a list of required technical skills as strings

Job Description:
{job_description[:15000]}
"""

# Load data
df = pd.read_json(INPUT_PATH, lines=True)

if args.test:
    df = df.head(5)
    OUTPUT_PATH = "data/processed/labeled_jobs_test.json"
    print(f"TEST MODE: running on {len(df)} rows only")
else:
    print(f"Total rows to process: {len(df)}")

# Resume support: skip already processed rows
processed = set()
if Path(OUTPUT_PATH).exists():
    existing = pd.read_json(OUTPUT_PATH, lines=True)
    processed = set(existing.index.tolist())
    print(f"Resuming  {len(processed)} rows already done")

os.makedirs("data/processed", exist_ok=True)

# Main loop 
output_file = open(OUTPUT_PATH, "a", encoding="utf-8")
errors = 0
MAX_RETRIES = 3 # Stop trying after 3 attempts per row

for idx, row in df.iterrows():
    if idx in processed:
        continue

    retries = 0
    success = False

    while retries < MAX_RETRIES and not success:# retry the CURRENT row max 3 times
        try:
            prompt   = build_prompt(row["job_description"])
            response = client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
            text     = response.text.strip()

            parsed = json.loads(text)

            record = {
                "title":                      row["title"],
                "source":                     row["source"],
                "job_description":            row["job_description"],
                "summary":                    parsed.get("summary", ""),
                "required_skills":            parsed.get("required_skills", [])
            }

            # json.dumps() converts the Python dictionary into a properly formatted JSON text string.
            output_file.write(json.dumps(record) + "\n")
            
            # .flush() forces Python to write the data to the hard drive IMMEDIATELY. 
            output_file.flush()

            if idx % 50 == 0:
                print(f"[{idx}/{len(df)}] processed: errors so far: {errors}")
            
            success = True 

        except json.JSONDecodeError:
            print(f"[{idx}] JSON parse error skipping")
            errors += 1
            break # don't retry if Gemini just gave us bad JSON
            
        except Exception as e:
            retries += 1
            print(f"[{idx}] Error: {e}: waiting 30s before retrying (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(30)
            #[577] Error: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}: waiting 30s before retrying (Attempt 1/3)
            
            if retries == MAX_RETRIES:
                print(f"[{idx}] Failed after {MAX_RETRIES} attempts. Skipping.")
                errors += 1

    time.sleep(DELAY)

output_file.close()

print(f"\nDone! Saved to {OUTPUT_PATH}")
print(f"Total errors: {errors}")