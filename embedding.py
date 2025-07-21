# embedding.py
"""
임베딩 생성·DB 저장 모듈
-------------------------------------------------
• 필요 라이브러리 : openai, pandas, tqdm, typing, logging
• DB 연동       : fetch.py의 DatabaseConnection 사용
• 스키마        : entity.py의 VectorEmbedding, Document, DocumentContent 참조
-------------------------------------------------
사용 예시
---------
from embedding import EmbeddingConfig, EmbeddingGenerator
from fetch import DatabaseConnection

# 1) DB에서 전처리된 청크 데이터 로드
db = DatabaseConnection()
# 예: id별로 processed_chunks 컬럼(JSON)까지 가져왔다고 가정
df = db.fetch_data("SELECT id, processed_chunks FROM your_table")

# 2) 임베딩 생성
cfg = EmbeddingConfig(model="text-embedding-3-small", batch_size=64)
gen = EmbeddingGenerator(cfg, db)
gen.process_dataframe(df, id_col="id", chunks_col="processed_chunks")
"""
import os
import json
import logging
from openai import OpenAI
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from fetch import DatabaseConnection
from entity import VectorEmbedding, DocumentContent, Document, engine
from sqlalchemy.orm import Session

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
            model: str = "text-embedding-3-small",
            batch_size: int = 32,
            openai_api_key: str | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------
# 2) 임베딩 생성기 클래스
# -------------------------------------------------
class EmbeddingGenerator:
    def __init__(self, config: EmbeddingConfig, db: DatabaseConnection):
        self.config = config
        self.db = db
        self.client = config.client

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API 호출로 텍스트 배치 임베딩 생성."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.config.model
        )
        return [item.embedding for item in response.data]

    def process_dataframe(
            self,
            df: pd.DataFrame,
            id_col: str = "id",
            chunks_col: str = "processed_chunks",
    ):
        """
        DataFrame을 순회하며 각 문서별 청크 임베딩 생성 후 DB 저장.

        df: id, processed_chunks 컬럼이 있어야 함.
        """
        session = Session(bind=engine)
        try:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="문서 임베딩"):
                # Document 테이블에 저장
                doc = Document(
                    title=row.get("title", "doc_{row[id_col]}"),
                    doc_type="past",
                    created_at=row["created_at"],
                    original_id=row[id]
                )
                session.add(doc)
                session.flush()  # Insert실행 -> doc.id를 가져오기 위한 작업

                chunks: List[Dict] = row[chunks_col]  # [{"chunk_index":0, "content":"...", ...}, ...]

                texts = [c["content"] for c in chunks]
                embeddings = []
                # 배치 사이즈 단위로 나누어 호출
                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i: i + self.config.batch_size]
                    embeddings.extend(self._embed_batch(batch))

                # DB에 저장
                for chunk, embed in zip(chunks, embeddings):
                    ve = VectorEmbedding(
                        document_id=doc.id,
                        chunk_index=chunk["chunk_index"],
                        content_preview=chunk["content_preview"],
                        token_count=chunk["token_count"],
                        embedding=embed,
                    )
                    session.add(ve)
                # (선택) 원본 청크 전체 텍스트 저장
                for chunk in chunks:
                    dc = DocumentContent(
                        document_id=doc.id,
                        chunk_index=chunk["chunk_index"],
                        full_content=chunk["content"],
                    )
                    session.add(dc)

                session.commit()
                logger.info(f"✅ 문서 {doc.id} 임베딩 및 저장 완료 (청크 {len(chunks)}개)")
        except Exception as e:
            session.rollback()
            logger.error(f"임베딩 처리 중 오류: {e}")
            raise
        finally:
            session.close()
