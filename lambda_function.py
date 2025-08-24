import logging
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from chain import MultiRouteChain
from pipeline import embedding_pipeline

timeline_chain = MultiRouteChain()

app = FastAPI()

logger = logging.getLogger("app")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(req: Request, exc: RequestValidationError):
    request_id = getattr(req.state, "request_id", "-")
    # Pydantic/FastAPI 유효성 오류 상세를 그대로 기록
    logger.warning(f"[{request_id}] 422 ValidationError: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id},
    )


class ChatRequest(BaseModel):
    query: str
    user_id: str
    locale: str


class EmbeddingRequest(BaseModel):
    user_id: str


@app.middleware("http")
async def log_requests(req: Request, call_next):
    start = time.perf_counter()

    # 요청 정보 상세 로깅
    logger.info(f"Request: {req.method} {req.url} - Headers: {dict(req.headers)}")

    if req.method == "POST":
        body = await req.body()
        content_type = req.headers.get("content-type", "")
        logger.info(f"Request body: {body} - Content-Type: {content_type}")

        # body를 다시 사용할 수 있도록 스트림 재설정
        async def receive():
            return {"type": "http.request", "body": body}

        req._receive = receive

    try:
        response = await call_next(req)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{req.method} {req.url.path} -> {response.status_code} ({duration_ms:.1f} ms)"
        )
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            f"Unhandled error during {req.method} {req.url.path} ({duration_ms:.1f} ms): {e}"
        )
        raise


@app.post("/chat")
def chat(
    request: ChatRequest,
    is_legacy: bool = Header(default=False, alias="is-legacy"),
):
    effective_user_id = os.environ.get("USER_ID") if is_legacy else request.user_id

    logger.info(f"/chat start legacy={is_legacy}, user_id={effective_user_id}")

    if not request.query or not effective_user_id:
        logger.warning("/chat validation failed: query or user_id is empty")
        raise HTTPException(status_code=400, detail="UserId or Query must not be empty")
    return StreamingResponse(
        timeline_chain.stream(request.query, effective_user_id, request.locale),
        media_type="text/event-stream",
    )


@app.post("/embedding")
def embedding(
    request: EmbeddingRequest,
    is_legacy: bool = Header(default=False, alias="is-legacy"),
):
    effective_user_id = os.environ.get("USER_ID") if is_legacy else request.user_id
    logger.info(f"/embedding start legacy={is_legacy}, user_id={effective_user_id}")

    if not effective_user_id:
        logger.warning("/embedding validation failed: user_id empty")
        raise HTTPException(status_code=400, detail="UserId must not be empty")

    try:
        embedding_pipeline(effective_user_id)
        logger.info(f"/embedding done user_id={effective_user_id}")

        return {"status": "embeddings generated"}
    except Exception as e:
        logger.exception(f"/embedding failed user_id={effective_user_id}")

        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
