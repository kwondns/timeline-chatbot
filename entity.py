import urllib
from datetime import datetime

from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    MetaData,
    Table,
    create_engine,
    event, Index, ForeignKey, Engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy.sql import func, expression
from pgvector.sqlalchemy import Vector, HALFVEC
import os

from sqlalchemy.sql.expression import text


# 1. SQLAlchemy Engine
def make_engine_from_env(options="?options=-c%20search_path%3Dtimeline_embedding") -> Engine:
    load_dotenv()
    pwd = urllib.parse.quote_plus(os.getenv('DB_PASSWORD', ''))
    url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{pwd}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}{options}"
    eng = create_engine(url)
    event.listen(eng, "connect", lambda con, _: register_vector(con))
    return eng


# 2. 엔진 생성
engine = make_engine_from_env()


@event.listens_for(engine, "connect")
def _register_vector(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


# 3. 메타데이터에 스키마 지정
metadata = MetaData(schema="timeline_embedding")
Base = declarative_base(metadata=metadata)


# 4. 테이블 정의
class Document(Base):
    """
    핵심 메타데이터만 보관하는 문서 테이블
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer,
        autoincrement=True,
        primary_key=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    original_id: Mapped[str] = mapped_column(String(100), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 역참조용 관계
    embeddings = relationship(
        "VectorEmbedding", back_populates="document", cascade="all, delete-orphan"
    )
    contents = relationship(
        "DocumentContent", back_populates="document", cascade="all, delete-orphan"
    )
    __table_args__ = (
        Index("idx_doc_type_created", "doc_type", "created_at"),
    )


class VectorEmbedding(Base):
    """
    1 청크 = 1 벡터. 쿼리용 프리뷰·토큰 수만 보관
    """
    __tablename__ = "vector_embeddings"

    id: Mapped[int] = mapped_column(
        Integer,
        autoincrement=True,
        primary_key=True,
    )
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_preview: Mapped[str] = mapped_column(String(200))
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # pgvector 필드: text-embedding-3-large → 3072차원
    embedding: Mapped[list[float]] = mapped_column(Vector(3072), nullable=False)

    document = relationship("Document", back_populates="embeddings")

    __table_args__ = (
        # ① 표현식 인덱스로 embedding을 halfvec(3072)로 캐스팅
        # ② operator class로 halfvec_cosine_ops(코사인) 또는 halfvec_l2_ops(L2) 지정
        Index(
            "idx_embedding_halfvec_hnsw",
            expression.cast(Column("embedding"), HALFVEC(3072)),
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={None: "halfvec_cosine_ops"}  # 또는 "halfvec_l2_ops"
        ),
        Index("idx_vec_chunk_unique", "document_id", "chunk_index", unique=True),
    )


class DocumentContent(Base):
    """
    대용량 원본 텍스트 저장 (TOAST 압축)
    """
    __tablename__ = "document_contents"

    id: Mapped[int] = mapped_column(
        Integer,
        autoincrement=True,
        primary_key=True,
    )
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    full_content: Mapped[str] = mapped_column(Text, nullable=False)

    document = relationship("Document", back_populates="contents")

    __table_args__ = (
        Index("idx_content_lookup", "document_id", "chunk_index", unique=True),
    )


# 5. 확장(vector) 활성화 및 스키마·테이블 생성
def initialize_database():
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {metadata.schema}'))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector with schema timeline_embedding'))
        Base.metadata.create_all(bind=conn)
    print("✅ DB 초기화 완료")


if __name__ == "__main__":
    initialize_database()
