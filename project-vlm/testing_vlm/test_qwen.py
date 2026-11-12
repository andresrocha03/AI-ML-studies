import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model_id = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Model loaded successfully!")
print("Device:", model.device)
print("CUDA available:", torch.cuda.is_available())
