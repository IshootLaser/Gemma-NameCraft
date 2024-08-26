import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apis.routes import health, chat_completion

root_path = "/"
if os.getenv("ROUTE"):
    root_path = os.getenv("ROUTE")

app = FastAPI(root_path=root_path)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat_completion.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18544)
