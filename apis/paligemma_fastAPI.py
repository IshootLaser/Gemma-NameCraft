from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from paligemma_inference import model_manager
from inference_utils import streaming_async, string_to_image

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


@app.post('/generate')
def generate(vqa: VQA):
    streamer = model_manager.get_streamer()
    t = Thread(target=model_manager.predict, args=(vqa.prompt, string_to_image(vqa.image), streamer))

    t.start()
    return StreamingResponse(streaming_async(streamer), media_type='text/event-stream')
