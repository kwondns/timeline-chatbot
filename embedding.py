import os
import logging
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from typing import List, Dict
from tqdm import tqdm
from fetch import DatabaseConnection

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -------------------------------------------------
# 1) 설정 객체
# -------------------------------------------------
class EmbeddingConfig:
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 32,
        openai_api_key: str | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAIEmbeddings(
            model=self.model, api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        db = DatabaseConnection()
        self.engine = db.get_engine()


# -------------------------------------------------
# 2) 임베딩 생성기 클래스
# -------------------------------------------------
class EmbeddingGenerator:
    def __init__(self, collection="past"):
        self.config = EmbeddingConfig()
        self.embeddings = self.config.client
        self.collection = collection
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            connection=self.config.engine,
            collection_name=f"{collection}_documents",
            pre_delete_collection=False,
            use_jsonb=True,
        )

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

            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk["content"],
                        metadata={
                            "chunk_index": chunk["chunk_index"],
                            "content_preview": chunk["content_preview"],
                            "token_count": chunk["token_count"],
                            "created_at": str(row["created_at"]),
                            "doc_id": str(row["id"]),
                            "doc_type": self.collection,
                            "title": row["title"],
                        },
                    )
                )
            self.vector_store.add_documents(documents)
            logger.info(
                f"✅ 문서 {row['id']} 임베딩 및 저장 완료 (청크 {len(chunks)}개)"
            )
