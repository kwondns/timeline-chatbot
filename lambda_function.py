import os
import json
from typing import Any, Dict
from chain import MultiRouteChain
from embedding import EmbeddingGenerator
from fetch import DatabaseConnection
from pipeline import embedding_pipeline

db_conn = DatabaseConnection()
timeline_chain = MultiRouteChain()
embedding_gen = EmbeddingGenerator()


def handler(event, context, response_stream):
    path = event.get("rawPath") or event.get("path", "")
    body = event.get("body", "{}")
    if isinstance(body, str):
        body = json.loads(body)

    if path.endswith("/chat"):
        query = body.get("query", "").strip()
        if not query:
            return _response(400, {"detail": "Query must not be empty"})
        response_stream.set_content_type("text/event-stream")
        for chunk in timeline_chain.stream(query):
            response_stream.write(json.dumps(chunk))
        response_stream.end()

    elif path.endswith("/embedding"):
        try:
            embedding_pipeline()
            return _response(200, {"status": "embeddings generated"})
        except Exception as e:
            return _response(500, {"detail": str(e)})
    else:
        return _response(404, {"detail": "Not Found"})


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
