import os
import zipfile
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# UNZIP MODEL (IMPORTANT)
# =========================
zip_path = "ai_model.zip"
extract_path = "ai_model"

if not os.path.exists(extract_path):
    print("Unzipping model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Model ready!")

# =========================
# BASE MODEL
# =========================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# =========================
# LOAD YOUR FINE-TUNED MODEL
# =========================
print("Loading LoRA model...")
model = PeftModel.from_pretrained(base_model, extract_path)

# =========================
# CHAT FUNCTION
# =========================
def chat(message, history):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# =========================
# GRADIO UI
# =========================
demo = gr.ChatInterface(
    fn=chat,
    title="My AI ChatGPT Clone",
    description="Custom trained AI model running online"
)

# =========================
# RUN SERVER (RENDER FIX)
# =========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000
    )
