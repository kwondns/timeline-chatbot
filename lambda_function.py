import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chain import MultiRouteChain
from pipeline import embedding_pipeline

timeline_chain = MultiRouteChain()

app = FastAPI()


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return StreamingResponse(
        timeline_chain.stream(request.query), media_type="text/event-stream"
    )


@app.post("/embedding")
def embedding():
    try:
        embedding_pipeline()
        return {"status": "embeddings generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
