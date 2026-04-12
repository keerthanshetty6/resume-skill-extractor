# Resume Skill Extractor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

An LLM fine-tuned to analyze job descriptions and extract required skills 
to help candidates tailor their resumes.

## What it does
Paste any job description → get a concise summary + list of required skills instantly.

## Example Output
> **Input:** Senior Data Scientist role at Veeva Systems focusing on NLP pipelines 
> and large-scale semantic analysis over medical documents...

**Output:**
```
Summary: This role involves designing an end-to-end pipeline for extracting 
information from large-scale unstructured medical documents using NLP and ML. 
The position requires building semantic search capabilities and working with 
LLMs on cloud infrastructure.

Required Skills: Python, NLP, PyTorch, Hugging Face Transformers, 
AWS/GCP/Azure, Docker, Kubernetes, Named Entity Recognition, 
Semantic Search, Large Language Models
```
*(Example from test run — full model coming soon)*

## Project Structure
```
resume-skill-extractor/
├── data/                          # Not tracked in git
│   ├── raw/
│   │   └── combined_jobs.json         # Combined raw job descriptions
│   └── processed/
│       ├── labeled_jobs.json          # Full Gemini-labeled dataset
│       ├── labeled_jobs_clean.json    # Cleaned dataset (3,050 rows)
│       ├── labeled_jobs_test.json     # Test run output (5 rows)
│       └── removed_rows.csv          # Removed junk rows for inspection
├── src/
│   ├── data_prep.py           # Downloads and combines job datasets
│   ├── generate_labels.py     # Uses Gemini API to extract skills
│   ├── labeled_jobs_clean.py  # Cleans and filters labeled dataset
│   └── available_models.py    # Lists available Gemini models
├── fine_tuning_pre_trained_LLM.py  # Fine-tuning script (in progress)
├── app.py                     # Gradio demo UI (coming soon)
├── requirements.txt
└── README.md
```

## Dataset
- 3,050 AI/Data Science job descriptions
- Sources: [nathansutton/data-science-job-descriptions](https://huggingface.co/datasets/nathansutton/data-science-job-descriptions) + [batuhanmtl/job-skill-set](https://huggingface.co/datasets/batuhanmtl/job-skill-set)
- Labels generated using Gemini 2.5 Flash Lite

## Tech Stack
- Python, PyTorch, Hugging Face Transformers
- PEFT / LoRA (parameter-efficient fine-tuning)
- Gemini 2.5 Flash Lite API (dataset labeling)
- Gradio (demo UI)

## Model
- Base model: Llama 3 (fine-tuned with LoRA)
- Training data: 3,050 labeled job descriptions
- Fine-tuning method: QLoRA (4-bit quantization)

## Status
- [x] Data collection and cleaning
- [x] Label generation with Gemini API
- [ ] Fine-tuning with LoRA
- [ ] Gradio demo

## Why I built this
Most job seekers struggle to tailor their resumes to specific roles. This tool 
instantly identifies the exact skills a job description is looking for, so 
candidates know what to highlight.

## Author
**Keerthan Shetty**  
[LinkedIn](https://www.linkedin.com/in/keerthanmshetty/) | [Hugging Face](https://huggingface.co/k10shetty)

## License
MIT License - feel free to use and build on this project.