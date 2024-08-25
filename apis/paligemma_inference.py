import gc
import os
from time import time, sleep

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, TextIteratorStreamer

# basic paligemma serving API, implemented following the official guide @
# https://huggingface.co/google/paligemma-3b-pt-896/tree/main
model_path = os.environ.get('MODEL_PATH', './models/paligemma')


class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.load_time = None
        self.working = False
        self.loading = False

    def load(self):
        self.loading = True
        t1 = time()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, device_map='cuda:0'
        ).eval()
        t2 = time()
        self.load_time = round(t2 - t1, 3)
        self.loading = False

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None

    def predict(self, prompt, image, max_tokens, sample, streamer):
        self.working = True
        if (model_manager.model is None) and not self.loading:
            self.load()
        else:
            while self.loading:
                sleep(0.001)
        model_inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(model_manager.model.device)
        with torch.no_grad():
            self.model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=sample, streamer=streamer)
        self.unload()
        self.working = False

    @staticmethod
    def get_streamer():
        return TextIteratorStreamer(model_manager.processor, skip_prompt=True, skip_special_tokens=True)


model_manager = ModelManager()
model_manager.load()
print(f'first model loading took {model_manager.load_time} seconds')
model_manager.unload()
