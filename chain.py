import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from zoneinfo import ZoneInfo

from langchain.retrievers import MultiQueryRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain.chains.query_constructor.base import AttributeInfo
from sqlalchemy.orm import sessionmaker

from config import Config
from fetch import DatabaseConnection
from route_logging_service import log_route_decision


class MultiRouteChain:
    def __init__(self):
        self.config = Config()
        db = DatabaseConnection()
        self.engine = db.get_engine()
        self.embeddings = self.config.embeddings
        self.llm = self.config.llm
        self.vectorstore = self.config.vector_store
        self.session_local = sessionmaker(bind=self.engine)

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
            AttributeInfo(name="user_id", description="사용자 식별자", type="string"),
        ]

        # 라우팅 체인 설정
        self._setup_routing_chain()
        self._setup_main_chain()

    def _log_route_decision(
        self,
        query: str,
        user_id: str,
        routing_result: Dict,
        locale: str,
        latency_ms: int,
        success: bool = True,
        error_message: str = None,
    ):
        try:
            with self.session_local() as session:
                confidence_score = None
                if routing_result.get("confidence") is not None:
                    confidence_score = int(routing_result["confidence"] * 100)
                log_route_decision(
                    session,
                    query_text=query,
                    route_name=routing_result["destination"],
                    user_id=user_id,
                    source="chat",
                    decision_score=confidence_score,
                    retriever=routing_result["destination"],
                    locale=locale,
                    model=self.config.model,
                    latency_ms=latency_ms,
                    success=success,
                    error_message=error_message,
                    extra={
                        "reasoning": routing_result.get("reasoning"),
                        "search_kwargs": routing_result.get("search_kwargs", {}),
                        "original_confidence": routing_result.get("confidence"),
                    },
                )
        except Exception as e:
            print(f"Error logging route decision: {e}")

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
        
        5. ETC: 과거에 작업한 내용이랑 상관 없는 단순 정보성 질문
            - "오늘이 며칠이지?" 혹은 "지금 몇시야"와 같은 단순 검색
            

        쿼리: 반환 예시
        1. 가장 마지막 React 작업은? -> WEIGHTED
        2. 가장 마지막으로 한 작업은? -> SORT
        3. 최근 GSAP 작업은? -> WEIGHTED
        
        JSON 형태로 응답하세요:
        {{
            "destination": "FILTER|WEIGHTED|SORT|SIMILARITY|ETC",
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
        """네 가지 검색 체인 설정"""

    def make_self_query_retriever(self, user_id: str) -> SelfQueryRetriever:
        # 1. FILTER: SelfQueryRetriever (조건부 필터링)
        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="사용자의 과거 작업 내용",
            metadata_field_info=self.metadata_field_info,
            verbose=True,
            search_kwargs={"k": 5, "filter": {"user_id": user_id}},
        )

    def make_weighted_retriever(self, user_id: str) -> TimeWeightedVectorStoreRetriever:
        # 2. WEIGHTED: TimeWeightedVectorStoreRetriever (시간 가중치)
        return TimeWeightedVectorStoreRetriever(
            vectorstore=self.vectorstore,
            decay_rate=0.999,
            k=5,
            memory_stream=self.config.get_memory(),
            search_kwargs={
                "filter": {
                    "user_id": user_id,
                    "created_at": {
                        "$gt": (datetime.now() - timedelta(days=30)).timestamp()
                    },
                }
            },
        )

    def make_similarity_retriever(
        self, user_id: str
    ) -> TimeWeightedVectorStoreRetriever:
        # 3. SIMILARITY: 기본 벡터 유사도 검색
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "filter": {"user_id": user_id}},
        )

    def make_multi_query_retriever(self, user_id: str) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(
            retriever=self.make_similarity_retriever(user_id),
            llm=self.llm,
            include_original=True,
        )

    def _direct_filter_vector_search(
        self, query: str, search_kwargs: Dict, user_id: str
    ) -> List:
        k = search_kwargs.get("k", 5)
        sort_criteria = search_kwargs.get("sort", [])
        if sort_criteria:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1000, "filter": {"user_id": user_id}},
            )
            docs = retriever.invoke(query)

            for sort in sort_criteria:
                if "created_at" in sort:
                    reverse = sort["created_at"] == -1
                    docs = sorted(
                        docs, key=lambda x: x.metadata["created_at"], reverse=reverse
                    )
            return docs[:k]
        return []

    def _setup_main_chain(self):
        """메인 LCEL 체인 구성"""

        def route_query(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """라우팅 결정 및 실행"""
            query = inputs["query"]
            user_id = inputs["user_id"]
            locale = inputs["locale"]

            routing_start_time = time.time()

            try:

                # 라우팅 결정
                routing_result = self.route_chain.invoke({"query": query})
                routing_latency_ms = int((time.time() - routing_start_time) * 1000)

                # 통계 업데이트
                destination = routing_result["destination"]

                # 검색기 선택 및 실행
                if destination == "FILTER":
                    docs = self.make_self_query_retriever(user_id).invoke(query)
                elif destination == "WEIGHTED":
                    docs = self.make_weighted_retriever(user_id).invoke(query)
                elif destination == "SORT":
                    search_kwargs = routing_result.get("search_kwargs", {})
                    if "sort" in search_kwargs:
                        docs = self._direct_filter_vector_search(
                            query, search_kwargs=search_kwargs, user_id=user_id
                        )
                elif destination == "SIMILARITY":
                    docs = self.make_multi_query_retriever(user_id).invoke(query)
                else:  # ETC
                    answer = self.config.llm.invoke(query)
                    self._log_route_decision(
                        query, user_id, routing_result, locale, routing_latency_ms, True
                    )

                    return {
                        "query": query,
                        "documents": [],
                        "routing_decision": destination,
                        "reasoning": routing_result["reasoning"],
                        "confidence": routing_result["confidence"],
                        "direct_answer": answer.content.strip(),
                        "locale": locale,
                    }
                self._log_route_decision(
                    query, user_id, routing_result, locale, routing_latency_ms, True
                )
                return {
                    "query": query,
                    "documents": docs,
                    "routing_decision": destination,
                    "reasoning": routing_result["reasoning"],
                    "confidence": routing_result["confidence"],
                    "locale": locale,
                }
            except Exception as e:
                routing_latency_ms = int((time.time() - routing_start_time) * 1000)

                # 실패 로깅
                error_routing_result = {
                    "destination": "ERROR",
                    "reasoning": "라우팅 실패",
                    "confidence": 0.0,
                }
                self._log_route_decision(
                    query,
                    user_id,
                    error_routing_result,
                    locale,
                    routing_latency_ms,
                    False,
                    str(e),
                )

                # 예외를 다시 발생시켜 상위에서 처리하도록 함
                raise

        def timestamp_to_kst(ts: float) -> str:
            return (
                datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
                .astimezone(ZoneInfo("Asia/Seoul"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )

        def generate_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """최종 답변 생성"""
            if "direct_answer" in inputs:
                # ETC: 오직 direct_answer만 출력
                yield inputs["direct_answer"]
                return
            query = inputs["query"]
            docs = inputs["documents"]
            locale = inputs["locale"]

            # 컨텍스트 구성
            context = "\n\n".join(
                [
                    f"제목: {doc.metadata.get('title', 'N/A')}\n"
                    f"생성일: {timestamp_to_kst(doc.metadata.get('created_at', 'N/A'))}\n"
                    f"내용: {doc.page_content}"
                    for doc in docs
                ]
            )

            answer_locale = {
                "ko": "위 정보를 바탕으로 상세하고 실용적인 답변을 한국어로 제공해주세요.",
                "en": "Please provide a detailed and practical answer in English based on the above information.",
                "ja": "上記の情報に基づいて、詳細で実用的な回答を日本語で提供してください。",
                "zh-cn": "请基于以上信息用中文提供详细且实用的回答。",
                "es": "Por favor proporciona una respuesta detallada y práctica en español basada en la información anterior.",
                "fr": "Veuillez fournir une réponse détaillée et pratique en français basée sur les informations ci-dessus.",
            }

            answer_locale_message = answer_locale.get(locale, answer_locale["ko"])
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

                {answer_locale_message}
                """,
                    ),
                ]
            )

            answer = answer_prompt | self.llm
            inner_stream = answer.stream(
                {
                    "query": query,
                    "context": context,
                    "answer_locale_message": answer_locale_message,
                }
            )

            for event in inner_stream:
                yield event.content  # 모델의 실제 출력만 전달

        # 메인 체인 구성
        self.main_chain = (
            RunnablePassthrough()
            | RunnableLambda(route_query)
            | RunnableLambda(generate_answer)
        )

    def stream(self, query: str, user_id: str, locale: str):
        """스트리밍 실행"""
        for chunk in self.main_chain.stream(
            {"query": query, "user_id": user_id, "locale": locale}
        ):
            yield chunk
