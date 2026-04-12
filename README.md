# Resume Skill Extractor

An LLM fine-tuned to analyze job descriptions and extract required skills 
to help candidates tailor their resumes.

## What it does
Paste any job description → get a concise summary + list of required skills instantly.

## Project Structure
resume-skill-extractor/
├── data/          # Raw and processed datasets (not tracked in git)
├── src/
│   ├── data_prep.py          # Downloads and combines job datasets
│   ├── generate_labels.py    # Uses Gemini API to extract skills
│   └── format_data.py        # Formats data for fine-tuning (coming soon)
├── models/        # Fine-tuned LoRA adapters (not tracked in git)
├── app.py         # Gradio demo UI (coming soon)
└── requirements.txt

## Dataset
- 3,050 AI/Data Science job descriptions
- Sources: nathansutton/data-science-job-descriptions + batuhanmtl/job-skill-set
- Labels generated using Gemini 2.5 Flash Lite

## Tech Stack
- Python, PyTorch, Hugging Face Transformers
- PEFT (LoRA fine-tuning)
- Gemini API (dataset labeling)
- Gradio (demo UI)

## Status
- [x] Data collection and cleaning
- [x] Label generation with Gemini API
- [ ] Fine-tuning with LoRA
- [ ] Gradio demo