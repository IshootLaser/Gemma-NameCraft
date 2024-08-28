from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse, Response, JSONResponse

from paligemma_inference import model_manager
from inference_utils import streaming, string_to_image

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VQA(BaseModel):
    prompt: str
    image: str
    max_tokens: int = 100
    sample: bool = False


def streaming_wrapper(func, thread, *args, **kwargs):
    for i in func(*args, **kwargs):
        yield i
    thread.join()
    yield ''


@app.post('/generate')
def generate(vqa: VQA):
    if model_manager.working:
        return Response('Model is already loaded, please wait for the current inference to finish.', 503)

    streamer = model_manager.get_streamer()
    t = Thread(
        target=model_manager.predict,
        args=(vqa.prompt, string_to_image(vqa.image), vqa.max_tokens, vqa.sample, streamer)
    )

    t.start()
    return StreamingResponse(streaming_wrapper(streaming, t, streamer), media_type='text/event-stream')


@app.get('/preload')
def preload():
    if (model_manager.model is None) and not model_manager.loading:
        model_manager.load()
        return Response('Model preloaded.', 200)
    else:
        return Response('Model is already loaded.', 200)


@app.get('/health')
def health():
    if model_manager.load_time is not None:
        return JSONResponse(content={'is_loaded': True})
    else:
        return JSONResponse(content={'is_loaded': False})
