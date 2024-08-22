import gc
import json

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, TextIteratorStreamer
from PIL import Image
import torch
from flask import Flask, request, Response, stream_with_context
import os
import base64
from io import BytesIO
from threading import Thread
from time import time
from datetime import datetime


# basic paligemma serving API, implemented following the official guide @
# https://huggingface.co/google/paligemma-3b-pt-896/tree/main
model_path = os.environ.get('MODEL_PATH', './models/paligemma')


class ModelManager:
    def __init__(self):
        self.model = None
        print(os.listdir(), model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.load_time = None

    def load(self):
        t1 = time()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, device_map='cuda:0'
        ).eval()
        t2 = time()
        self.load_time = round(t2 - t1, 3)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


model_manager = ModelManager()
model_manager.load()
print(f'first model loading took {model_manager.load_time} seconds')
model_manager.unload()


def predict(prompt, image, streamer):
    model_manager.load()
    model_inputs = model_manager.processor(
        text=prompt, images=image, return_tensors="pt"
    ).to(model_manager.model.device)
    with torch.no_grad():
        model_manager.model.generate(**model_inputs, max_new_tokens=100, do_sample=True, streamer=streamer)
    model_manager.unload()


app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    image = Image.open(BytesIO(base64.b64decode(data['image']))).convert('RGB')
    streamer = TextIteratorStreamer(model_manager.processor, skip_prompt=True, skip_special_tokens=True)
    t = Thread(target=predict, args=(prompt, image, streamer))

    def streaming():
        chunk = {
            'payload': '',
            'finished': False,
            'last_load_time': model_manager.load_time
        }
        chunk = f'data: {json.dumps(chunk)}\n\n'
        yield chunk
        for i in streamer:
            chunk = {
                'payload': i,
                'finished': False,
                'last_load_time': model_manager.load_time
            }
            chunk = f'data: {json.dumps(chunk)}\n\n'
            yield chunk
        chunk = {
            'payload': '',
            'finished': True,
            'last_load_time': model_manager.load_time
        }
        chunk = f'data: {json.dumps(chunk)}\n\n'
        yield chunk

    t.start()
    response = Response(stream_with_context(streaming()), content_type='text/event-stream')
    response.headers['Transfer-Encoding'] = 'chunked'
    response.headers['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
    return response


app.run('0.0.0.0', 5023)
