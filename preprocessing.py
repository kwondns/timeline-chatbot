import logging
import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -------------------------------------------------
# 1) 설정 객체
# -------------------------------------------------
@dataclass
class PreprocessingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    language: str = "korean"  # "english" 등 확장 가능


# -------------------------------------------------
# 2) 전처리기 클래스
# -------------------------------------------------
class TextPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # ---------- 단계별 전처리 ----------
    @staticmethod
    def clean_text(text: str) -> str:
        """HTML·URL·특수문자 제거 및 공백 정리."""
        text = re.sub(r"<[^>]+>", " ", text)  # HTML 태그
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            text,
        )  # URL
        text = re.sub(
            r"[^\w\s\u3131-\u3163\uac00-\ud7a3\u4e00-\u9fff.,!?;:()\[\]{}\"\'\-]",
            " ",
            text,
        )  # 특수문자
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def korean_preprocessing(text: str) -> str:
        """한국어 특화 정규화."""
        text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", text)  # 자모 단독 제거
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        return text.strip()

    # ---------- 청킹 ----------
    def chunk_text(self, text: str) -> List[Dict]:
        if self.text_splitter:
            raw_chunks = self.text_splitter.split_text(text)
        else:
            raw_chunks = self.simple_chunk_text(text)

        chunk_data: List[Dict] = []
        for idx, chunk in enumerate(raw_chunks):
            chunk = chunk.strip()
            chunk_data.append(
                {
                    "chunk_index": idx,
                    "content": chunk,
                    "content_preview": (
                        (chunk[:197] + "...") if len(chunk) > 200 else chunk
                    ),
                    "token_count": self.calculate_tokens(chunk),
                    "char_count": len(chunk),
                }
            )
        return chunk_data

    # ---------- 토큰 계산 ----------
    def calculate_tokens(self, text: str) -> int:
        """대략적 토큰 수 추정 (OpenAI 기준)."""
        if self.config.language.lower() == "korean":
            return int(len(text) * 0.7)
        return int(len(text.split()) * 1.3)

    # ---------- 공개 API ----------
    def preprocess_document(self, document: str) -> List[Dict]:
        """문서 단위 전처리 실행."""

        clean = (
            self.korean_preprocessing(self.clean_text(document))
            if self.config.language == "korean"
            else self.clean_text(document)
        )
        chunks = self.chunk_text(clean)
        logger.info(f"✅ 전처리 완료: {len(chunks)}개 청크 생성")
        return chunks

    def get_statistics(self, chunks: List[Dict]) -> Dict:
        total_tokens = sum(c["token_count"] for c in chunks)
        total_chars = sum(c["char_count"] for c in chunks)
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
        }

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        source_col: str = "content",
        output_col: str = "processed_chunks",
    ) -> pd.DataFrame:
        """
        DataFrame의 source_col 텍스트를 전처리하여,
        output_col에 각 행별 chunks 리스트를 추가한 새 DataFrame 반환.
        """
        df = df.copy()

        def _process(row):
            raw = row[source_col]
            return self.preprocess_document(raw)

        df[output_col] = df.apply(_process, axis=1)
        return df
