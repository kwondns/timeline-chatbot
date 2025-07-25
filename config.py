import os
import pickle
import boto3

from io import BytesIO

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from fetch import DatabaseConnection


class Config:
    def __init__(self, collection_name="past"):
        load_dotenv()
        db = DatabaseConnection()
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-large"
        self.temperature = 0.1
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.engine = db.get_engine()
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model, api_key=self.openai_api_key
        )
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            connection=self.engine,
            collection_name=f"{collection_name}_documents",
            pre_delete_collection=False,
            use_jsonb=True,
        )

        self.memory_path = "/mnt/efs/time_weighted_memory_stream.pkl"
        self.bucket = os.getenv("S3_BUCKET")
        self.key = os.getenv("S3_KEY")
        self.s3 = boto3.client("s3")

    def _upload_memory_s3(self, memory):
        buffer = BytesIO()
        pickle.dump(memory, buffer)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=self.key, Body=buffer.read())
        print(f"S3://{self.bucket}/{self.key} Memory Upload Complete")

    def _get_memory_s3(self):
        from botocore.exceptions import ClientError

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            memory = pickle.loads(response["Body"].read())
            return memory
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return []
            elif error_code == "NoSuchBucket":
                return []
            else:
                print(
                    f"S3에서 파일을 가져오는 중 오류 발생: {e}. 빈 리스트를 반환합니다."
                )
                return []
        except Exception as e:
            print(f"메모리 로드 중 예상치 못한 오류 발생: {e}. 빈 리스트를 반환합니다.")
            return []

    def _upload_memory_local(self, memory):
        with open(self.memory_path, "wb") as f:
            pickle.dump(memory, f)

    def _get_memory_local(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "rb") as f:
                return pickle.load(f)
        else:
            return []

    def upload_memory(self, memory):
        env = os.getenv("ENVRIONMENT")
        if env == "development":
            return self._upload_memory_local(memory)
        elif env == "production":
            return self._upload_memory_s3(memory)

    def get_memory(self):
        env = os.getenv("ENVRIONMENT")
        if env == "development":
            return self._get_memory_local()
        elif env == "production":
            return self._get_memory_s3()
