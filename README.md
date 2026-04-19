# Resume Skill Extractor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An 8-Billion parameter Llama 3 model fine-tuned with QLoRA to analyze job descriptions and instantly extract required skills helping candidates tailor their resumes to specific roles.

## Demo

![Demo](Media/demo.gif)

## What it does
Paste any job description → get a concise summary + list of required skills instantly.

## Example Output

**Input:** Data Engineering role at smartclip, Berlin processing terabytes of TV and advertising data...

**Output:**
```
Summary: This Data Engineering role involves processing and refining terabytes 
of TV and advertising data, designing and building scalable data pipelines for 
TV and forecasting products, and ensuring pipelines are fast, clean, and reliable.

Required Skills:
- Python
- SQL
- Spark
- Hadoop
- Testing
- CI/CD
```

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
│       └── removed_rows.csv           # Removed junk rows for inspection
├── Media/
│   ├── demo.gif               # Demo fig file 
│   └── training_curves.png    
├── models/                    # Not tracked in git
├── src/
│   ├── data_prep.py           # Downloads and combines job datasets
│   ├── generate_labels.py     # Uses Gemini API to extract skills
│   ├── labeled_jobs_clean.py  # Cleans and filters labeled dataset
│   ├── inspect_lengths.py     # Analyses token lengths of dataset
│   ├── available_models.py    # Lists available Gemini models
│   ├── upload_dataset.py      # Pushes clean dataset to Hugging Face Hub
│   ├── merge_local.py         # Merge the base model with the adapters
|   └── train.py               # QLoRA fine-tuning with Llama 3
├── wandb/                     # Not tracked in git
├── app.py                     # Gradio demo UI 
├── requirements.txt
└── README.md
```

## How it was built

### 1. Dataset
- Combined 3,050 AI/Data Science job descriptions from two sources
- Used **Gemini 2.5 Flash Lite API** to automatically generate labels (summary + required skills) for each row
- Hosted on Hugging Face: [keerthanshetty/resume-skill-extractor-dataset](https://huggingface.co/datasets/keerthanshetty/resume-skill-extractor-dataset)

### 2. Fine-Tuning
- Base model: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Method: **QLoRA** 4-bit NF4 quantization + LoRA rank 16, alpha 32
- Target modules: all attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) + feed-forward layers (`gate_proj`, `up_proj`, `down_proj`)
- Hardware: NVIDIA A10G (24GB VRAM) on Lightning AI
- Training: 3 epochs, ~1,000 steps loss reduced from 2.2 → 1.0

### 3. Models on Hugging Face
- LoRA Adapters: [keerthanshetty/resume-skill-extractor-lora](https://huggingface.co/keerthanshetty/resume-skill-extractor-lora)
- Merged Model: [keerthanshetty/resume-skill-extractor-merged](https://huggingface.co/keerthanshetty/resume-skill-extractor-merged)

## Tech Stack
- **Fine-tuning:** PyTorch, Hugging Face Transformers, PEFT, TRL
- **Dataset labeling:** Gemini 2.5 Flash Lite API
- **Experiment tracking:** Weights & Biases
- **Demo UI:** Gradio + WordCloud + Matplotlib
- **Training platform:** Lightning AI (A10G GPU)

## Training Curves

| Metric | Start | End |
|--------|-------|-----|
| Train Loss | 2.2 | ~1.0 |
| Token Accuracy | 55% | ~75% |

![Training Curves](Media/training_curves.png)

## Status
- ✅ Data collection and cleaning
- ✅ Label generation with Gemini API
- ✅ Fine-tuning with QLoRA (3 epochs, A10G GPU)
- ✅ Gradio demo UI with word cloud

## Why I built this
Most job seekers struggle to tailor their resumes to specific roles. This tool 
instantly identifies the exact skills a job description is looking for, so 
candidates know exactly what to highlight.

## Author
**Keerthan Shetty**  
[LinkedIn](https://www.linkedin.com/in/keerthanmshetty/) | [Hugging Face](https://huggingface.co/keerthanshetty)

## License
MIT License - feel free to use and build on this project.