import os
import requests
from time import sleep
from base64 import b64encode
from PIL import Image
from io import BytesIO
import socket

server_url = os.environ.get('SERVER_URL', '127.0.0.1')
infinity_server = 'http://' + server_url + ':7997/health'
gemma_server = 'http://' + server_url + ':11434/v1/chat/completions'
paligemma_server = 'http://' + server_url + ':5443/generate'
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
image = Image.open('./test_img.jpg')
with BytesIO() as output:
    image.save(output, format='JPEG')
    image_b64 = b64encode(output.getvalue()).decode('utf-8')
paligemma_test_payload = {
    'prompt': 'caption en',
    'image': image_b64
}

max_retry = 10
retry_interval = 60


def check():
    r = requests.get(infinity_server)
    assert r.status_code == 200, f'Infinity server is not available at {infinity_server}'

    r = requests.post(gemma_server, json=gemma_test_payload)
    assert r.status_code == 200, f'Gemma server is not available at {gemma_server}'

    r = requests.post(paligemma_server, json=paligemma_test_payload)
    assert r.status_code == 200, f'Paligemma server is not available at {paligemma_server}'

    with socket.create_connection((server_url, 5432), timeout=5):
        pass


for n in range(max_retry):
    try:
        check()
        print('All services are available.')
        break
    except Exception as e:
        print(e)
        sleep(retry_interval)
        print(f'Number of retries remaining: {n}')
