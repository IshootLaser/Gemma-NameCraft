import asyncio
import base64
from io import BytesIO
from PIL import Image
import json
from paligemma_inference import model_manager


def string_to_image(b64_string):
    image = Image.open(BytesIO(base64.b64decode(b64_string))).convert('RGB')
    return image


def streaming(s):
    chunk = {
        'payload': '',
        'finished': False,
        'last_load_time': model_manager.load_time
    }
    chunk = f'data: {json.dumps(chunk)}\n\n'
    yield chunk
    for i in s:
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


async def streaming_async(s):
    chunk = {
        'payload': '',
        'finished': False,
        'last_load_time': model_manager.load_time
    }
    chunk = f'data: {json.dumps(chunk)}\n\n'
    yield chunk
    for i in s:
        chunk = {
            'payload': i,
            'finished': False,
            'last_load_time': model_manager.load_time
        }
        chunk = f'data: {json.dumps(chunk)}\n\n'
        yield chunk
        await asyncio.sleep(1)
    chunk = {
        'payload': '',
        'finished': True,
        'last_load_time': model_manager.load_time
    }
    chunk = f'data: {json.dumps(chunk)}\n\n'
    yield chunk
