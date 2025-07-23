import os
import logging
import pickle

import pandas as pd

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime

from fetch import DatabaseConnection
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class EmbeddingGenerator:
    def __init__(self, collection="past"):
        self.config = Config()
        self.embeddings = self.config.embeddings
        self.collection = collection
        self.vector_store = self.config.vector_store
        self.memory_path = self.config.memory_path

        if os.path.exists(self.memory_path):
            with open(self.memory_path, "rb") as f:
                saved_data = pickle.load(f)
        else:
            saved_data = []

        self.time_weighted_vector_store_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store,
            decay_rate=0.005,
            time_key="created_at",
            memory_stream=saved_data,
        )

    def _to_ts(self, ts):  # pandas.Timestamp, datetime, str 모두 처리
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ts.timestamp() if isinstance(ts, datetime) else float(ts)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        chunks_col: str = "processed_chunks",
    ):
        """
        DataFrame을 순회하며 각 문서별 청크 임베딩 생성 후 DB 저장.

        df: id, processed_chunks 컬럼이 있어야 함.
        """

        for df_index, row in tqdm(df.iterrows(), total=len(df), desc="문서 임베딩"):
            documents: List[Document] = []

            # [{"chunk_index":0, "content":"...", ...}, ...]
            chunks: List[Dict] = row[chunks_col]
            created_ts = self._to_ts(row["created_at"])
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk["content"],
                        metadata={
                            "chunk_index": chunk["chunk_index"],
                            "content_preview": chunk["content_preview"],
                            "token_count": chunk["token_count"],
                            "created_at_str": str(row["created_at"]),
                            "created_at": created_ts,
                            "last_accessed_at": created_ts,
                            "doc_id": str(row["id"]),
                            "doc_type": self.collection,
                            "title": row["title"],
                        },
                    )
                )
            """
            PGVector가 아닌 TimeWeightedVectorStoreRetriever로 문서등록
            해당 객체로 add_documents를 해야 메모리에 시간 가중치에 대한 값들도 적재되어 검색 가능
            """
            self.time_weighted_vector_store_retriever.add_documents(documents)
            logger.info(
                f"✅ 문서 {row['id']} 임베딩 및 저장 완료 (청크 {len(chunks)}개)"
            )
        with open(self.memory_path, "wb") as f:
            pickle.dump(self.time_weighted_vector_store_retriever.memory_stream, f)
