from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch
from flask import Flask, request, jsonify
import os
import base64
from io import BytesIO

# basic paligemma serving API, implemented following the official guide @
# https://huggingface.co/google/paligemma-3b-pt-896/tree/main

model_path = os.environ.get(
    'MODEL_PATH', './models/models--google--paligemma-3b-mix-224/snapshots/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515'
)
device = 'cuda:0'
dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path, quantization_config=quantization_config
).eval()
processor = AutoProcessor.from_pretrained(model_path)
app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    image = Image.open(BytesIO(base64.b64decode(data['image']))).convert('RGB')
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    return jsonify({'response': decoded})


app.run('0.0.0.0', 5023)
