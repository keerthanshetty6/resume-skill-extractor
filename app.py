import gradio as gr
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import numpy as np


# Load the AI Model
print("Loading model and adapters")

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# You can also change this to "models/fine_tuned_adapters" if you want to use your local folder
ADAPTER_PATH = "k10shetty/resume-skill-extractor-lora" 

# We must load the model in 4-bit, exactly as we did during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Merge our custom LoRA brain onto the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("Model loaded successfully!")

# The AI Generation Function
def extract_skills(job_description):
    if not job_description.strip():
        return "Please enter a job description.", "", None
    
    # A. Format the prompt EXACTLY like we did in our training script
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical recruiter. Analyze job descriptions and extract key information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Analyze this job description and provide a summary and required skills:

{job_description[:6500]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert technical recruiter. Analyze job descriptions and extract key information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Analyze this job description and provide a summary, the required skills, and the bonus/nice-to-have skills:

{job_description[:6500]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    # B. Send text to the GPU
    # pt" creates PyTorch Tensors. .to(model.device) safely moves the text to whatever hardware the model is on (GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    
    # C. Generate the answer
    outputs = model.generate(**inputs,          # Unpacks the text tokens and attention mask automatically
                            max_new_tokens=400, # Hard limit of ~300 words so the GPU doesn't get stuck in an infinite loop
                            temperature=0.1,    # Low temperature -> less creative
                            do_sample=True      # MUST be True for temperature to work. Allows probability-based word selection.
                            )    

    # outputs[0] grabs the first response from the batch, skip_special_tokens=True deletes the Llama 3 <|eot_id|> tags.
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # D. Parse the output 
    # The model repeats the prompt before answering, so we split it at the assistant header
    try:
        answer_only = full_response.split("assistant")[-1].strip()
        
        # Split Summary and Skills based on the bold headers we trained it on
        parts = answer_only.split("**Required Skills:**")
        summary_text = parts[0].replace("**Summary:**", "").strip()
        skills_raw = parts[1].strip() if len(parts) > 1 else "Skills format not found."

        # Clean up the asterisk bullets into a readable list
        if skills_raw != "Skills format not found.":
            skills_lines = [
                line.strip().lstrip("*").strip() 
                for line in skills_raw.split("\n") 
                if line.strip() and line.strip() != "*"
            ]
            # Keep headers (like **Bonus Skills:**) without bullets, but bullet the actual skills
            skills_text = "\n".join(
                s if s.startswith("**") else f"• {s}" 
                for s in skills_lines if s
            )
        else:
            skills_text = skills_raw
            
    except Exception as e:
        summary_text = "Error parsing model output."
        skills_text = full_response # Fallback to raw text if parsing fails

        
    # E. Generate the Visual Word Cloud

    # 1. Create a mathematical oval mask so the cloud isn't a boring rectangle
    # x is a column vector (400 rows, 1 column), y is a row vector (1 row, 800 columns).
    # np.ogrid is memory-efficient because it doesn’t create full 2D arrays.
    x, y = np.ogrid[:400, :800]
    # Apply the equation of an ellipse:
    # ((x - 200)/200)^2 + ((y - 400)/400)^2 > 1
    # Center of ellipse → (200, 400)
    # Radii → 200 (vertical), 400 (horizontal)
    # Points where the expression is > 1 lie OUTSIDE the ellipse.
    # This creates a boolean mask: True outside, False inside.
    mask = ((x - 200)/200)**2 + ((y - 400)/400)**2 > 1
    # Convert boolean mask to integer values:
    # True → 1, False → 0
    # Then scale to 255 so it can be used as an image mask:
    # 255 (white) = masked/ignored area (outside ellipse)
    # 0 (black) = visible area (inside ellipse)
    mask = 255 * mask.astype(int)

    # 2. Build the visually appealing cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='#1f2937',  # Sleek dark gray/blue that blends with Gradio's dark mode
        colormap='viridis',          # A professional gradient of blues, teals, and greens
        mask=mask,                   # Applies our smooth oval shape
        max_words=80,                # Limits the cloud to the top 80 words to prevent ugly clutter
        prefer_horizontal=0.85,      # Forces 85% of words to be horizontal so it is easy to read
        stopwords=STOPWORDS,
        contour_width=2,             # Adds a subtle outline to the oval
        contour_color='#374151'      # The color of the outline
    ).generate(job_description)
    
    fig, ax = plt.subplots(figsize=(8, 4))

    # Optional: Set the matplotlib figure background to match so there are no white borders
    fig.patch.set_facecolor('#1f2937')

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return summary_text, skills_text, fig


# The UI (Interface) 
# gr.Interface is the easiest way to build a UI. We just link our function and define the boxes.
demo = gr.Interface(
    fn=extract_skills,          # The function we defined above

    # The Inputs 
    inputs=gr.Textbox(
        lines=12, 
        placeholder="Paste job description here...", 
        label="Job Description"
        ),

    # The Outputs
    # Because our function returns two strings and 1 image, we need a list of two Textboxes here and a Plot to catch the 'fig' we returned above!
    outputs=[
        gr.Textbox(label="AI Summary", lines=3),
        gr.Textbox(label="Required Skills", lines=6),
        gr.Plot(label="Keyword Word Cloud")
    ],
    title="🚀 Resume Skill Extractor",
    description="Powered by a custom-trained Llama 3 8B model. Paste any job description to instantly extract the core responsibilities and necessary skills.",
    examples=[
        ["We are looking for a Senior Data Scientist to join our ML team. You will build predictive models using Python, PyTorch and SQL. Experience with AWS and Docker required. Knowledge of NLP and transformer architectures is a plus."],
    ],
    flagging_mode="never"                   # Hides the default "Flag" button Gradio puts on apps
)

# Launch!
if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch(share=True) # share=True creates a public public web link!