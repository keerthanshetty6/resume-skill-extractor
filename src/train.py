import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig #,TrainingArguments
from trl import SFTTrainer, SFTConfig #Transformers Reinforcement Learning
import torch
import wandb
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model #Parameter-Efficient Fine-Tuning

if torch.cuda.is_available():
    print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print(" WARNING: No GPU detected. Training will be slow.")

load_dotenv()

# Args 
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run on 20 rows only")
args = parser.parse_args()

# Config 
HF_TOKEN       = os.environ["HF_TOKEN"]
BASE_MODEL     = "meta-llama/Meta-Llama-3-8B-Instruct"  # it offers the best ratio of high intelligence to low VRAM usage, and it's already pre-trained to follow instructions. High intelligence, low VRAM, instruction-tuned
DATASET_ID     = "k10shetty/resume-skill-extractor-dataset" #Dataset on huggingface
OUTPUT_DIR     = "models/fine_tuned_adapters" #Saves ONLY the tiny LoRA weights, not the base model
MAX_SEQ_LEN    = 2048 #Caps GPU memory usage to prevent Out-Of-Memory (OOM) crashes

# Format each row into an instruction prompt
def format_prompt(row):
    # Converting the JSON list of skills into a clean bulleted text list
    skills = "\n".join(f"* {s}" for s in row["required_skills"])
    # Using official Meta Llama 3 special tokens: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical recruiter. Analyze job descriptions and extract key information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Analyze this job description and provide a summary and required skills:

{row['job_description'][:6500]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
**Summary:** {row['summary']}

**Required Skills:**
{skills}<|eot_id|>"""
# Note: We slice the job description at [:6500] characters (~1600 tokens) 
# to leave room in our 2048 MAX_SEQ_LEN for the assistant's answer.

# Load dataset
print(f"Loading dataset directly from Hugging Face: {DATASET_ID}...")

dataset = load_dataset(DATASET_ID, split="train")

if args.test:
    dataset = dataset.select(range(5))
    print(f"TEST MODE: {len(dataset)} rows")
else:
    print(f"Full dataset: {len(dataset)} rows")

# Split 90% train, 10% eval
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

# Apply prompt formatting to every row
# This creates a new column called "text" which the SFTTrainer will look for natively
train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)}) #adds a new column text for each row
eval_dataset  = eval_dataset.map(lambda x:  {"text": format_prompt(x)}) #adds a new column text for each row
"""
LoRA (The Math Trick): Instead of retraining a model's 8 Billion parameters (which requires massive supercomputers), we freeze the model completely. 
We then inject tiny, parallel "sticky notes" (Matrices A and B) into the network. These tiny matrices compress and decompress the data (Rank Decomposition), 
allowing us to train only ~29 million parameters (0.36% of the model) while achieving the exact same learning results.

QLoRA (The Memory Trick): To fit the massive frozen model onto a single, standard GPU, we compress it from 16-bit precision down to 4-bit precision, shrinking its size by 75%. 
During training, the GPU quickly unzips a few 4-bit weights into 16-bit, does the math, and throws them away, keeping your VRAM usage incredibly low."""

# Quantization config (QLoRA)
# This compresses the massive 8 Billion parameter base model so it fits on our GPU.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Shrinks base model weights from 16-bit to 4-bit
    bnb_4bit_quant_type="nf4",               # Sets how the 16 buckets are spaced (0000-1111). "nf4" spaces them to match the bell curve of AI weights, preventing accuracy loss.
    bnb_4bit_compute_dtype=torch.bfloat16,   # Temporarily unzips weights to 16-bit for math. bf16 sacrifices tiny fractional precision for massive range, preventing crashes.
    bnb_4bit_use_double_quant=True           # Compresses the quantization scaling factors to save even more VRAM, 64 weight blocks to 4 bit, scaling factor 32 bit. scaling factor to 8 bit under double quant (Compresses the 32-bit scaling factors down to 8-bit, saving ~400MB of VRAM across the model.)
)

# Load base model + tokenizer
print(f"Loading base model: {BASE_MODEL}")

# Load the exact dictionary (Tokenizer) used by Llama 3
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN) 

#Llama 3 doesn't have a default "blank space" token. We set it to the "End of Sentence" token.
tokenizer.pad_token = tokenizer.eos_token # This prevents crashes when the GPU batches rows of different lengths together.

# Important for Llama 3 training stability: Add the blank spaces to the END of the text,
# so the actual text position always starts cleanly at index 0.
tokenizer.padding_side = "right" 

# Load the base model applying our 4-bit compression rules
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto", # Automatically figures out the best way to load onto the GPU (or CPU fallback)
    token=HF_TOKEN
)

# Prepares the model for PEFT/LoRA:

# Locks the 4-bit weights(model weight), stabilizes LayerNorms to 32-bit, and enables gradient checkpointing
model = prepare_model_for_kbit_training(model)

# LoRA config
# The blueprint for our PEFT (Parameter-Efficient Fine-Tuning) adapters
lora_config = LoraConfig(
    r=16,                                       # Rank: The "brain capacity" of the adapter. 16 is the sweet spot for complex formatting tasks.
    lora_alpha=32,                              # Scaling factor: Multiplies LoRA output by (alpha/rank) -> (32/16) = 2. Gives the new weights a louder voice.
    target_modules=[                            # Which specific linear layers inside the Transformer we attach adapters to:
        "q_proj", "k_proj", "v_proj", "o_proj", # The Attention Mechanism
        "gate_proj", "up_proj", "down_proj"     # The Feed-Forward Network (MLP)
    ],
    lora_dropout=0.05,                          # Drops 5% of LoRA weights randomly during training to prevent memorization (overfitting).
    bias="none",                                # Ignores bias variables (Wx+b) to save VRAM. (Can be "lora_only" if needed).
    task_type="CAUSAL_LM"                       # Tells the library we are training a text-generation model (predicts the next word).
)

model = get_peft_model(model, lora_config)      # Injects our LoRA adapter blueprint into the model

model.print_trainable_parameters()

# Training args
# The control center for how the model learns, uses memory, and saves data.
# NEW IN TRL 1.1.0: SFTConfig replaces TrainingArguments and absorbs the dataset/formatting args.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",          # Tells the trainer to look at the "text" column we created earlier
    max_length=MAX_SEQ_LEN,             # Enforces the 2048 token limit to prevent memory crashes, Renamed from max_seq_length
    num_train_epochs=3,                               
    per_device_train_batch_size=2,      # 2 rows at a time to prevent VRAM Out-of-Memory crashes.
    gradient_accumulation_steps=4,      # Accumulate math over 4 steps to simulate a stable batch size of 8.
    learning_rate=2e-4,                 
    bf16=True,                          # Use bfloat16 for training stability on modern Ampere GPUs.
    logging_steps=10,                   # Send metrics to Weights & Biases every 10 steps.
    eval_strategy="epoch",              # Test the model on the holdout evaluation dataset at the end of every epoch
    save_strategy="epoch",              # Save a checkpoint of the LoRA adapters at the end of every epoch.
    load_best_model_at_end=True,        # Automatically load the smartest checkpoint at the end, not just the last one.
    report_to="wandb",                  # Route all logs to the WandB cloud dashboard.
    run_name="resume-skill-extractor-full-run"
)

# Train
# Open a connection to Weights & Biases to log live training metrics
wandb.init(project="resume-skill-extractor")

# The SFTTrainer automates the entire complex PyTorch training loop
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,     # Renamed from tokenizer
    args=training_args,
)

print("Starting training...")
trainer.train() # This executes the actual math! (Will take hours depending on hardware)

# Save LoRA adapters 
# Because we used PEFT, this safely saves ONLY the ~60MB of adapter weights, not the 16GB base model.
trainer.model.save_pretrained(OUTPUT_DIR)

# Save the tokenizer so our custom padding rules are preserved for the Gradio app
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapters saved to {OUTPUT_DIR}")

# Push adapters to Hugging Face
trainer.model.push_to_hub("k10shetty/resume-skill-extractor-lora", token=HF_TOKEN)
tokenizer.push_to_hub("k10shetty/resume-skill-extractor-lora", token=HF_TOKEN)
print("Pushed to Hugging Face!")