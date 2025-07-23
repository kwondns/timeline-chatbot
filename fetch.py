import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import urllib

load_dotenv()


class DatabaseConnection:
    def __init__(self):
        self.database = os.getenv("DB_DATABASE")
        self.user = os.getenv("DB_USER")
        self.password = urllib.parse.quote(os.getenv("DB_PASSWORD"), safe="")
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.schema = os.getenv("DB_SCHEMA")

        if not all([self.database, self.user, self.password, self.host, self.port]):
            raise ValueError(
                "데이터베이스 연결 정보가 완전하지 않습니다. 환경변수를 확인해주세요."
            )

        self.engine = None

    def get_engine(self):
        if self.engine is None:
            connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?options=-c%20search_path%3D{self.schema}"
            self.engine = create_engine(connection_string)
        return self.engine

    def fetch_data(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql(query, con=self.get_engine())
        except SQLAlchemyError as e:
            print(f"데이터베이스 쿼리 실행 중 오류 발생: {e}")
            raise
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            raise

    def close_connection(self):
        if self.engine:
            self.engine.dispose()
