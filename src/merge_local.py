import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# ── 0. Environment Setup ───────────────────────────────────────────────────
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found! Please check your .env file.")

# ── 1. Config & Defensive Hardware Checks ──────────────────────────────────
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_REPO = "k10shetty/resume-skill-extractor-lora"
OUTPUT_DIR = "./merged-model"

# ✅ Safer dtype fallback (prevents rare crashes on older GPUs like T4)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Defend against VRAM fragmentation
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True

# ── 2. Load Base Model (Force full GPU placement & Prevent RAM Spikes) ─────
print(f"Loading base model in {dtype}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map=None,              # IMPORTANT: no auto-sharding
    trust_remote_code=True,
    token=hf_token,               # Required for gated Llama 3
    low_cpu_mem_usage=True        # Prevents System RAM OOM crashes
).to("cuda")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)

# ── 3. Load LoRA ───────────────────────────────────────────────────────────
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=hf_token)

# ── 4. Merge (The Safe Way) ────────────────────────────────────────────────
print("Merging model...")

model.eval()                      # Locks weights to a deterministic state

with torch.no_grad():             # Prevents backprop VRAM buffer spikes
    model = model.merge_and_unload()

# Force PyTorch to release lingering memory before the next step
del base_model
torch.cuda.empty_cache()

# ── 5. Move to CPU & Save Locally ──────────────────────────────────────────
print("Moving to CPU to avoid GPU OOM during serialization...")
model = model.to("cpu")

print(f"Saving locally to {OUTPUT_DIR} in 2GB chunks...")
model.save_pretrained(
    OUTPUT_DIR, 
    safe_serialization=True,
    max_shard_size="2GB"          # ✅ Prevents upload network failures
)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Local save complete! You can now turn off the GPU and upload from a CPU.")