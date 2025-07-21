# db.py
import os, urllib.parse as ul

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector, HALFVEC
from sqlalchemy import create_engine, Column, Integer, Text, JSON, MetaData, text, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import expression

def create_engine_from_env():
    load_dotenv()
    pwd = ul.quote_plus(os.getenv("DB_PASSWORD", ""))
    url = (
        "postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{pwd}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
        f"/{os.getenv('DB_DATABASE')}?options=-csearch_path%3Dtimeline_embedding"
    )
    return create_engine(url, pool_pre_ping=True)
engine = create_engine_from_env()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
metadata = MetaData(schema="timeline_embedding")
Base = declarative_base(metadata=metadata)

class TimelineDocument(Base):
    __tablename__ = "timeline_documents"
    id = Column(Integer, primary_key=True)
    document_id = Column(Text)
    content = Column(Text, nullable=False)
    meta= Column("metadata", JSON, default=dict)
    embedding = Column(Vector(3072), nullable=False)
    __table_args__ = (
        # ① 표현식 인덱스로 embedding을 halfvec(3072)로 캐스팅
        # ② operator class로 halfvec_cosine_ops(코사인) 또는 halfvec_l2_ops(L2) 지정
        Index(
            "idx_embedding_vector_halfvec_hnsw",
            expression.cast(Column("embedding"), HALFVEC(3072)),
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={None: "halfvec_cosine_ops"}  # 또는 "halfvec_l2_ops"
        ),
    )


def initialize_database():
    with engine.begin() as conn:
        # 스키마 생성
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS timeline_embedding'))
        # vector 확장 활성화 (스키마 지정)
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA timeline_embedding'))
        Base.metadata.create_all(bind=conn)
    print("DB 초기화")


if __name__ == "__main__":
    initialize_database()
