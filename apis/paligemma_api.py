from threading import Thread

from flask import Flask, request, Response, stream_with_context

from inference_utils import string_to_image, streaming
from paligemma_inference import model_manager

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    image = string_to_image(data['image'])
    streamer = model_manager.get_streamer()
    t = Thread(target=model_manager.predict, args=(prompt, image, streamer))

    t.start()
    response = Response(stream_with_context(streaming(streamer)), content_type='text/event-stream; charset=utf-8')
    return response


app.run('0.0.0.0', 5023)
