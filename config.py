import os
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
