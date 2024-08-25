import os
import requests
from time import sleep
from base64 import b64encode
from PIL import Image
from io import BytesIO
import socket
from pathlib import Path

infinity_server = os.environ.get('INFINITY_SERVER', 'http://embedding_services:7997/health')
gemma_server = os.environ.get('GEMMA_SERVER', 'http://ollama:11434/v1/chat/completions')
paligemma_server = os.environ.get('PALIGEMMA_SERVER', 'http://paligemma:5023/health')
postgres_host = os.environ.get('POSTGRES_HOST', 'postgres')
gemma_test_payload = {
    "model": "gemma2-2b-Chinese",
    "stream": False,
    "max_tokens": 1,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}
image = Image.open(os.path.join(Path(__file__).parent.absolute(), 'test_img.jpg'))
with BytesIO() as output:
    image.save(output, format='JPEG')
    image_b64 = b64encode(output.getvalue()).decode('utf-8')
paligemma_test_payload = {
    'prompt': 'caption en',
    'image': image_b64,
    'max_tokens': 1
}

max_retry = 10
retry_interval = 60


def check():
    service_ready = {
        'infinity': False,
        'gemma': False,
        'paligemma': False,
        'database': False
    }

    try:
        r = requests.get(infinity_server)
        if r.status_code == 200:
            service_ready['infinity'] = True
    except:
        pass

    try:
        r = requests.post(gemma_server, json=gemma_test_payload)
        if r.status_code == 200:
            service_ready['gemma'] = True
    except:
        pass

    try:
        r = requests.get(paligemma_server)
        if r.status_code == 200:
            service_ready['paligemma'] = True
    except:
        pass

    try:
        with socket.create_connection((postgres_host, 5432), timeout=5):
            service_ready['database'] = True
    except:
        pass

    return service_ready


if __name__ == '__main__':
    for n in range(max_retry):
        try:
            for k, v in check().items():
                if not v:
                    raise Exception(f'{k} service not available.')
        except Exception as e:
            print(e)
            sleep(retry_interval)
            print(f'Number of retries remaining: {n}')
