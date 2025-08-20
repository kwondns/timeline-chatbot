import os

import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chain import MultiRouteChain
from pipeline import embedding_pipeline

timeline_chain = MultiRouteChain()

app = FastAPI()


class ChatRequest(BaseModel):
    query: str
    user_id: str


class EmbeddingRequest(BaseModel):
    user_id: str


@app.post("/chat")
def chat(
    request: ChatRequest,
    is_legacy: bool = Header(default=False, alias="is-legacy"),
):
    effective_user_id = os.environ.get("USER_ID") if is_legacy else request.user_id

    if not request.query or not effective_user_id:
        raise HTTPException(status_code=400, detail="UserId or Query must not be empty")
    return StreamingResponse(
        timeline_chain.stream(request.query, effective_user_id),
        media_type="text/event-stream",
    )


@app.post("/embedding")
def embedding(
    request: EmbeddingRequest,
    is_legacy: bool = Header(default=False, alias="is-legacy"),
):
    effective_user_id = os.environ.get("USER_ID") if is_legacy else request.user_id

    if not effective_user_id:
        raise HTTPException(status_code=400, detail="UserId must not be empty")

    try:
        embedding_pipeline(effective_user_id)
        return {"status": "embeddings generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
