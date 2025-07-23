import os
import json
from typing import Any, Dict
from chain import MultiRouteChain
from embedding import EmbeddingGenerator
from fetch import DatabaseConnection

db_conn = DatabaseConnection()
timeline_chain = MultiRouteChain()
embedding_gen = EmbeddingGenerator()


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    path = event.get("rawPath") or event.get("path", "")
    body = json.loads(event.get("body", "{}"))

    if path.endswith("/chat"):
        query = body.get("query", "").strip()
        if not query:
            return _response(400, {"detail": "Query must not be empty"})
        result = timeline_chain.invoke(query)
        return _response(
            200,
            {
                "answer": result["answer"],
            },
        )

    elif path.endswith("/embedding"):
        data = body.get("data")
        if not isinstance(data, list):
            return _response(400, {"detail": "Data list required"})
        # 예: [{ "id": ..., "content": ... }, ...]
        try:
            embedding_gen.process_dataframe(
                df=_to_dataframe(data), chunks_col="processed_chunks"
            )
            return _response(200, {"status": "embeddings generated"})
        except Exception as e:
            return _response(500, {"detail": str(e)})

    else:
        return _response(404, {"detail": "Not Found"})


def _to_dataframe(data_list):
    import pandas as pd

    return pd.DataFrame(data_list)


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
