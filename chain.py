import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Any
from zoneinfo import ZoneInfo

from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain.chains.query_constructor.base import AttributeInfo

from config import Config
from fetch import DatabaseConnection

import langchain

langchain.debug = True if os.getenv("ENVIRONMENT") == "dev" else False


# 라우팅 스키마 정의 (사용자의 실제 요구사항 반영)
class TimelineRouteQuery(TypedDict):
    destination: Literal["FILTER", "WEIGHTED", "SIMILARITY"]
    reasoning: str
    confidence: float


class MultiRouteChain:
    """사용자의 실제 timeline 데이터에 최적화된 LCEL MultiRouteChain"""

    def __init__(self):
        self.config = Config()
        db = DatabaseConnection()
        self.engine = db.get_engine()
        self.routing_stats = {"FILTER": 0, "WEIGHTED": 0, "SORT": 0, "SIMILARITY": 0}
        self.embeddings = self.config.embeddings
        self.llm = self.config.llm
        self.vectorstore = self.config.vector_store

        # 실제 데이터 스키마에 맞는 메타데이터 필드 정의
        self.metadata_field_info = [
            AttributeInfo(
                name="title", description="작업 또는 문서의 제목", type="string"
            ),
            AttributeInfo(
                name="created_at",
                description=f"작업이 생성된 날짜와 시간 Unix timestamp float형태 (2024-02-01:{datetime(2024, 2, 1).timestamp()}, 2024-03-01:{datetime(2024, 3, 1).timestamp()})",
                type="float",
            ),
            AttributeInfo(
                name="doc_type", description="문서 타입 (주로 'past')", type="string"
            ),
            AttributeInfo(
                name="token_count", description="문서의 토큰 수", type="integer"
            ),
            AttributeInfo(
                name="created_at_str",
                description="작업이 생성된 날짜와 시간 문자열 타입",
                type="string",
            ),
        ]

        # 라우팅 체인 설정
        self._setup_routing_chain()
        self._setup_retrieval_chains()
        self._setup_main_chain()

    def _setup_routing_chain(self):
        """사용자의 과거 작업 특성에 맞춘 라우팅 시스템"""
        route_system = """
        사용자의 과거 작업 이력을 검색하는 최적의 방식을 한가지 선택하세요.

        데이터 특성:
        - 과거 작업한 내용들
        - 시간순으로 정리된 작업 이력

        라우팅 규칙:
        1. FILTER: 조건부 검색이 필요한 경우
           - 특정 날짜/기간: "2024년 2월", "최근 한 달"
           - 특정 기술/주제: "GSAP", "마크다운", "타임라인"
           - 키워드 기반 필터링

        2. WEIGHTED: 의미 유사도 + 최신성 가중치가 모두 중요할 때
            - 주제 키워드(예: “React”, “마크다운”) + 시간 개념(“최근”, “가장 최신”, “요즘”)이 혼합된 경우
            - "업데이트", "최근에 한", "요즘"
            - 시간 가중치가 필요한 검색
            - 특정 기술/ 주제가 있을 경우 해당 키워드가 있는 문서를 필터링 한 후 문서 검색
            - 주제 중심 검색에 최신 문서 우선 순위가 필요할 때
            - 명시적 날짜는 없으나 최신성이 핵심일 때 
        3. SORT: 최신성 단어만 있고 주제 키워드가 **전혀 없을 때만** 선택
            - 시간 키워드만 있을 경우(“가장 마지막”, “가장 오래된”, “가장 최신”)
            - 주제 키워드 없이 포괄적으로 시간을 기준으로 정렬할 때
            - search_kwargs를 생성하여 반환
            - sort -> creaetd_at 기준 내림차순 혹은 오름차순 정렬 [{{"created_at": -1}}]형태로 생성 (내림차순/-1 또는 오름차순/1)
            - k -> 반환할 문서 수

        4. SIMILARITY: 의미적 유사도 검색
           - 개념적 질문: "CSS 스타일링 방법"
           - 기술 설명: "애니메이션 구현"
           - 일반적인 지식 검색

        쿼리: 반환 예시
        1. 가장 마지막 React 작업은? -> WEIGHTED
        2. 가장 마지막으로 한 작업은? -> SORT
        3. 최근 GSAP 작업은? -> WEIGHTED
        
        JSON 형태로 응답하세요:
        {{
            "destination": "FILTER|WEIGHTED|SORT|SIMILARITY",
            "reasoning": "선택한 이유",
            "confidence": 0.0-1.0
            "search_kwargs": {{{{
             SORT로 구분되었을 때만 retriever.invoke에 그대로 넘길 search_kwargs를 생성
            }}}}
        }}
        """

        route_prompt = ChatPromptTemplate.from_messages(
            [("system", route_system), ("human", "질문: {query}")]
        )

        self.route_chain = route_prompt | self.llm | JsonOutputParser()

    def _setup_retrieval_chains(self):
        """세 가지 검색 체인 설정"""

        # 1. FILTER: SelfQueryRetriever (조건부 필터링)
        self.filter_retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="사용자의 과거 작업 내용",
            metadata_field_info=self.metadata_field_info,
            verbose=True,
        )

        # 2. WEIGHTED: TimeWeightedVectorStoreRetriever (시간 가중치)
        self.memory_path = self.config.memory_path
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "rb") as f:
                saved_data = pickle.load(f)
        else:
            saved_data = []
        self.weighted_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vectorstore,
            decay_rate=0.999,
            k=5,
            memory_stream=saved_data,
            search_kwargs={
                "filter": {
                    "created_at": {
                        "$gt": (datetime.now() - timedelta(days=30)).timestamp()
                    }
                }
            },
        )

        # Filter Retriever를 추가하여 Ensemble로 검색
        # self.filter_weighted_retriever = self.vectorstore.as_retriever(
        #     search_kwargs={
        #         "filter": {
        #             "created_at": {
        #                 "$gt": (datetime.now() - timedelta(days=30)).timestamp()
        #             }
        #         }
        #     }
        # )
        # self.ensemble = EnsembleRetriever(
        #     retrievers=[self.filter_weighted_retriever, self.weighted_retriever],
        #     weights=[0.5, 0.5],
        # )

        # self.multi_query_weighted_retriever = MultiQueryRetriever.from_llm(
        #     retriever=self.weighted_retriever,
        #     llm=self.llm,
        #     include_original=True,
        # )

        # 3. SIMILARITY: 기본 벡터 유사도 검색
        self.similarity_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        self.multi_query_similarity_retriever = MultiQueryRetriever.from_llm(
            retriever=self.similarity_retriever,
            llm=self.llm,
            include_original=True,
        )

    def _direct_filter_vector_search(self, query: str, search_kwargs: Dict) -> List:
        k = search_kwargs.get("k", 5)
        sort_criteria = search_kwargs.get("sort", [])
        if sort_criteria:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 1000,
                },
            )
            docs = retriever.invoke(query)

            for sort in sort_criteria:
                if "created_at" in sort:
                    reverse = sort["created_at"] == -1
                    docs = sorted(
                        docs, key=lambda x: x.metadata["created_at"], reverse=reverse
                    )
            return docs[:k]

    def _setup_main_chain(self):
        """메인 LCEL 체인 구성"""

        def route_query(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """라우팅 결정 및 실행"""
            query = inputs["query"]

            # 라우팅 결정
            routing_result = self.route_chain.invoke({"query": query})

            # 통계 업데이트
            destination = routing_result["destination"]
            self.routing_stats[destination] += 1

            # 검색기 선택 및 실행
            if destination == "FILTER":
                docs = self.filter_retriever.invoke(query)
            elif destination == "WEIGHTED":
                docs = self.weighted_retriever.invoke(query)
            elif destination == "SORT":
                search_kwargs = routing_result.get("search_kwargs", {})
                if "sort" in search_kwargs:
                    docs = self._direct_filter_vector_search(
                        query, search_kwargs=search_kwargs
                    )
            else:  # SIMILARITY
                docs = self.multi_query_similarity_retriever.invoke(query)

            return {
                "query": query,
                "documents": docs,
                "routing_decision": destination,
                "reasoning": routing_result["reasoning"],
                "confidence": routing_result["confidence"],
            }

        def timestamp_to_kst(ts: float) -> str:
            return (
                datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
                .astimezone(ZoneInfo("Asia/Seoul"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )

        def generate_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """최종 답변 생성"""
            query = inputs["query"]
            docs = inputs["documents"]

            # 컨텍스트 구성
            context = "\n\n".join(
                [
                    f"제목: {doc.metadata.get('title', 'N/A')}\n"
                    f"생성일: {timestamp_to_kst(doc.metadata.get('created_at', 'N/A'))}\n"
                    f"내용: {doc.page_content}"
                    for doc in docs
                ]
            )

            # 답변 생성 프롬프트
            answer_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                당신은 사용자의 과거 작업 이력을 분석하는 전문 어시스턴트입니다.
                제공된 과거 작업 내용을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

                답변 시 다음을 포함하세요:
                0. 관련된 작업들의 생성일자 
                1. 관련된 과거 작업들의 요약
                2. 구체적인 기술이나 방법론
                3. 작업 시점과 맥락
                4. 실용적인 조언이나 참고사항
                """,
                    ),
                    (
                        "human",
                        """
                질문: {query}

                관련 과거 작업들:
                {context}

                위 정보를 바탕으로 상세하고 실용적인 답변을 제공해주세요.
                """,
                    ),
                ]
            )

            answer = answer_prompt | self.llm
            result = answer.invoke({"query": query, "context": context})

            return {**inputs, "answer": result.content, "source_count": len(docs)}

        # 메인 체인 구성
        self.main_chain = (
            RunnablePassthrough()
            | RunnableLambda(route_query)
            | RunnableLambda(generate_answer)
        )

    def invoke(self, query: str) -> Dict[str, Any]:
        """동기 실행"""
        return self.main_chain.invoke({"query": query})

    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """비동기 실행"""
        return await self.main_chain.ainvoke({"query": query})

    def stream(self, query: str):
        """스트리밍 실행"""
        for chunk in self.main_chain.stream({"query": query}):
            yield chunk

    def get_routing_stats(self) -> Dict[str, Any]:
        """라우팅 통계 반환"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return self.routing_stats

        return {
            **self.routing_stats,
            "percentages": {
                k: round(v / total * 100, 2) for k, v in self.routing_stats.items()
            },
            "total_queries": total,
        }


# 실제 사용 예시
def main():
    """실제 사용 예시"""

    # 체인 초기화
    timeline_chain = MultiRouteChain()

    # 다양한 질문 유형 테스트
    test_queries = [
        "2024년 2월에 GSAP 관련해서 뭘 작업했나요?",  # → FILTER
        "가장 최근에 작업한 마크다운 스타일링 내용이 뭔가요?",  # → WEIGHTED
        "CSS 애니메이션 구현하는 방법이 뭐가 있나요?",  # → SIMILARITY
        "타임라인 관련 과거 작업들을 찾아주세요",  # → FILTER
        "가장 최근에 작업한 React에 대한 내용은 뭐야?",
        "가장 최근에 작업한 내용들을 요약해줘",
        "가장 마지막 React 작업은 뭐였지?",
        "가장 마지막 작업 내용",
        "Next에서 캐싱을 다루는 작업은 언제했지?",
    ]

    print("🎯 Timeline MultiRouteChain 테스트 시작\n")

    for i, query in enumerate(test_queries, 1):
        print(f"{'=' * 60}")
        print(f"테스트 {i}: {query}")
        print(f"{'=' * 60}")

        try:
            result = timeline_chain.invoke(query)

            print(f"📍 라우팅: {result['routing_decision']}")
            print(f"🤔 이유: {result['reasoning']}")
            print(f"📊 신뢰도: {result['confidence']:.2f}")
            print(f"📚 소스 개수: {result['source_count']}")
            print(f"\n💬 답변:")
            print(result["answer"])
            print(f"\n{'=' * 60}\n")

        except Exception as e:
            print(f"❌ 오류 발생: {e}\n")

    # 통계 출력
    stats = timeline_chain.get_routing_stats()
    print("📈 라우팅 통계:")
    for route, count in stats.items():
        if route != "percentages" and route != "total_queries":
            percentage = stats.get("percentages", {}).get(route, 0)
            print(f"  {route}: {count}회 ({percentage}%)")
    print(f"  총 쿼리: {stats.get('total_queries', 0)}회")


if __name__ == "__main__":
    main()
