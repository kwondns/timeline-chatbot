# Python
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, Session

from config import Config

Base = declarative_base()


class RouteLog(Base):
    __tablename__ = "timeline_chat_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # 추적 관련
    user_id = Column(String(128), nullable=True)  # 필요 시 마스킹하여 저장
    source = Column(String(32), nullable=True)  # "chat", "embedding" 등 호출 출처

    # 라우팅/쿼리 정보
    query_text = Column(Text, nullable=False)  # 원 쿼리 텍스트
    route_name = Column(String(64), nullable=False)  # 선택된 라우트 이름
    decision_score = Column(
        Integer, nullable=True
    )  # 점수형(정수/가중치가 실수면 Float로 변경)
    retriever = Column(String(64), nullable=True)  # 사용된 리트리버 이름
    model = Column(String(64), nullable=True)  # 사용 모델 이름
    locale = Column(String(64), nullable=True)
    latency_ms = Column(Integer, nullable=True)  # 라우팅 결정에 걸린 시간(ms)

    # 결과/오류 및 메타데이터
    success = Column(Boolean, nullable=False, server_default="true")
    error_message = Column(Text, nullable=True)
    extra = Column(JSONB, nullable=True)  # 추가 메타(예: 각 라우트별 점수 맵)

    __table_args__ = (
        Index("ix_route_logs_created_at", "created_at"),
        Index("ix_route_logs_route_name_created_at", "route_name", "created_at"),
    )


def create_route_logs_table(engine) -> None:
    """
    최초 1회 테이블 생성용. Alembic을 사용 중이라면 마이그레이션으로 대체하세요.
    """
    Base.metadata.create_all(bind=engine, tables=[RouteLog.__table__])


def log_route_decision(
    session: Session,
    *,
    query_text: str,
    route_name: str,
    user_id: Optional[str] = None,
    source: Optional[str] = None,
    decision_score: Optional[int] = None,  # 점수가 실수면: Optional[float]
    retriever: Optional[str] = None,
    model: Optional[str] = None,
    locale: str,
    latency_ms: Optional[int] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    """
    라우팅 결정 직후 호출하여 로그를 저장합니다. 커밋까지 수행하며 생성된 로그 ID를 반환합니다.
    """
    row = RouteLog(
        user_id=user_id,
        source=source,
        query_text=query_text,
        route_name=route_name,
        decision_score=decision_score,
        retriever=retriever,
        model=model,
        locale=locale,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
        extra=extra,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row.id


if __name__ == "__main__":
    config = Config()
    engine = config.engine
    create_route_logs_table(engine)
