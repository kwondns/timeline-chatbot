import os

from dotenv import load_dotenv

from embedding import EmbeddingGenerator
from fetch import DatabaseConnection
from preprocessing import PreprocessingConfig, TextPreprocessor

load_dotenv()


def embedding_pipeline():
    # 1. 데이터베이스 연결 및 데이터 가져오기
    print("데이터베이스 연결 중...")
    db = DatabaseConnection()

    lastEmbeddingQuery = """SELECT *
                            FROM timeline_embedding.langchain_pg_embedding
                            ORDER BY (cmetadata ->>'created_at_str')::timestamp DESC
                            LIMIT 1; \
                         """
    last_df = db.fetch_data(lastEmbeddingQuery)

    if last_df.empty:
        # (1) 임베딩 이력이 없음 → 전체 데이터 조회
        query = """
                SELECT *
                FROM timeline.past
                ORDER BY "startTime"; \
                """
        print("임베딩 이력이 없으므로 전체 데이터를 불러옵니다.")
    else:
        # (2) 최근 임베딩 이후 데이터만 조회
        last_time = last_df["cmetadata"][0]["created_at_str"]
        print(f"마지막 임베딩 시각: {last_time}")
        query = f"""
            SELECT *
            FROM timeline.past
            WHERE "startTime" > '{last_time}'
            ORDER BY "startTime";
        """

    df = db.fetch_data(query)
    print(f"데이터 {len(df)}개 로드 완료")
    if len(df) == 0:
        return
    # 2. 데이터 전처리
    print("데이터 전처리 중...")
    preprocessing_config = PreprocessingConfig()
    text_preprocessor = TextPreprocessor(preprocessing_config)
    df = text_preprocessor.preprocess_dataframe(df)

    print(f"전처리 완료: {len(df['content'])}개 레코드")

    # 3. 임베딩 생성 및 DB 저장
    print("임베딩 생성 중...")
    embedding_generator = EmbeddingGenerator()
    embedding_generator.process_dataframe(df, chunks_col="processed_chunks")

    # 4. 연결 종료
    db.close_connection()
    print("임베딩 파이프라인 완료!")


if __name__ == "__main__":
    embedding_pipeline()
