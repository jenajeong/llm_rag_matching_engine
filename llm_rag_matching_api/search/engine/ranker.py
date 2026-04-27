"""
Professor Ranker
AHP 기반 교수 순위 평가 모듈
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .ahp_config import (
    DEFAULT_TYPE_WEIGHTS,
    TIME_WEIGHTS,
    ARTICLE_CONTRIBUTION_WEIGHTS,
    ARTICLE_SCALE_WEIGHTS,
    PATENT_STATUS_WEIGHTS,
    PROJECT_SCALE_WEIGHTS,
    MAX_ARTICLE_SCORE_PER_DOC,
    MAX_PATENT_SCORE_PER_DOC,
    MAX_PROJECT_SCORE_PER_DOC,
    map_article_contribution,
    map_article_journal_type,
    map_patent_status,
    map_project_budget,
    calculate_time_weight
)
from .professor_aggregator import ProfessorAggregator


class ProfessorRanker:
    """교수 순위 평가 클래스"""
    
    def __init__(
        self,
        aggregator: Optional[ProfessorAggregator] = None
    ):
        """
        초기화
        
        Args:
            aggregator: 교수 집계기 (None이면 자동 생성)
        """
        self.aggregator = aggregator or ProfessorAggregator()
        self.current_year = datetime.now().year
    
    def rank_professors(
        self,
        professor_data: Dict[str, Dict[str, Any]],
        ahp_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        교수별 AHP 점수 계산 및 순위 매기기
        
        Args:
            professor_data: 교수별 집계 데이터 (ProfessorAggregator.aggregate_by_professor() 결과)
            ahp_weights: AHP 가중치 (None이면 기본값 사용)
                {
                    "patent": 0.2,
                    "article": 0.2,
                    "project": 0.4
                }
            
        Returns:
            순위가 매겨진 교수 리스트
            [
                {
                    "rank": 1,
                    "professor_id": "...",
                    "professor_info": {...},
                    "total_score": 0.85,
                    "scores_by_type": {
                        "patent": 0.35,
                        "article": 0.30,
                        "project": 0.20
                    },
                    "documents": {...},
                    "document_scores": {
                        "patent": [...],
                        "article": [...],
                        "project": [...]
                    }
                },
                ...
            ]
        """
        if ahp_weights is None:
            ahp_weights = DEFAULT_TYPE_WEIGHTS
        
        ranked_professors = []
        
        for prof_id, prof_data in professor_data.items():
            # 각 데이터 타입별 점수 계산
            scores_by_type = {}
            document_scores = {
                "patent": [],
                "article": [],
                "project": []
            }
            
            # Patent 점수 계산
            patent_docs = prof_data["documents"].get("patent", [])
            patent_scores = []
            for doc in patent_docs:
                doc_score = self._calculate_patent_score(doc)
                patent_scores.append(doc_score)
                document_scores["patent"].append({
                    "no": doc.get("no"),
                    "title": doc.get("title"),
                    "score": doc_score,
                    "metadata": doc.get("metadata", {})
                })
            # 타입별 점수: 합계 후 유형별 최대점으로 정규화 (논문/특허/연구과제 스케일 통일)
            # 정규화 없으면 특허 1건 점수가 논문 1건보다 2~3배 커서, 특허만 있어도 논문만 있는 교수보다 순위가 올라가는 문제 발생
            raw_patent = sum(patent_scores)
            scores_by_type["patent"] = raw_patent / MAX_PATENT_SCORE_PER_DOC if MAX_PATENT_SCORE_PER_DOC else raw_patent
            
            # Article 점수 계산
            article_docs = prof_data["documents"].get("article", [])
            article_scores = []
            for doc in article_docs:
                doc_score = self._calculate_article_score(doc)
                article_scores.append(doc_score)
                document_scores["article"].append({
                    "no": doc.get("no"),
                    "title": doc.get("title"),
                    "score": doc_score,
                    "metadata": doc.get("metadata", {})
                })
            raw_article = sum(article_scores)
            scores_by_type["article"] = raw_article / MAX_ARTICLE_SCORE_PER_DOC if MAX_ARTICLE_SCORE_PER_DOC else raw_article
            
            # Project 점수 계산
            project_docs = prof_data["documents"].get("project", [])
            project_scores = []
            for doc in project_docs:
                doc_score = self._calculate_project_score(doc)
                project_scores.append(doc_score)
                document_scores["project"].append({
                    "no": doc.get("no"),
                    "title": doc.get("title"),
                    "score": doc_score,
                    "metadata": doc.get("metadata", {})
                })
            raw_project = sum(project_scores)
            scores_by_type["project"] = raw_project / MAX_PROJECT_SCORE_PER_DOC if MAX_PROJECT_SCORE_PER_DOC else raw_project
            
            # 종합 점수: 유형 가중치 × 정규화된 타입별 점수 (동일 품질 1건당 article 0.4, patent 0.2, project 0.4 비율 반영)
            total_score = (
                ahp_weights.get("patent", 0) * scores_by_type["patent"] +
                ahp_weights.get("article", 0) * scores_by_type["article"] +
                ahp_weights.get("project", 0) * scores_by_type["project"]
            )
            
            ranked_professors.append({
                "professor_id": prof_id,
                "professor_info": prof_data["professor_info"],
                "total_score": total_score,
                "scores_by_type": scores_by_type,
                "documents": prof_data["documents"],
                "document_scores": document_scores
            })
        
        # 점수 기준 내림차순 정렬
        ranked_professors.sort(key=lambda x: x["total_score"], reverse=True)
        
        # 순위 추가
        for rank, prof in enumerate(ranked_professors, 1):
            prof["rank"] = rank
        
        return ranked_professors
    
    def _calculate_article_score(self, doc: Dict) -> float:
        """
        논문 문서의 AHP 점수 계산
        L1(시간) × L2(기여도) × L3(규모)
        
        Args:
            doc: 논문 문서 데이터
            
        Returns:
            점수 (0~1)
        """
        # L1: 시간 가중치
        year = doc.get("year")
        if not year:
            return 0.0
        
        time_key = calculate_time_weight(year, self.current_year)
        time_weight = TIME_WEIGHTS.get(time_key, 0.0)
        
        # L2: 기여도 가중치
        metadata = doc.get("metadata", {})
        contribution_value = metadata.get("THSS_PATICP_GBN", "")
        contribution_key = map_article_contribution(contribution_value)
        contribution_weight = ARTICLE_CONTRIBUTION_WEIGHTS.get(contribution_key, 0.0)
        
        # L3: 규모 가중치 (학술지 등급)
        journal_value = metadata.get("JRNL_GBN", "")
        journal_key = map_article_journal_type(journal_value)
        scale_weight = ARTICLE_SCALE_WEIGHTS.get(journal_key, 0.0)
        
        # 최종 점수 = 시간 × 기여도 × 규모
        score = time_weight * contribution_weight * scale_weight
        
        return score
    
    def _calculate_patent_score(self, doc: Dict) -> float:
        """
        특허 문서의 AHP 점수 계산
        L1(시간) × L4(권리상태)
        
        Args:
            doc: 특허 문서 데이터
            
        Returns:
            점수 (0~1)
        """
        # L1: 시간 가중치
        year = doc.get("year")
        if not year:
            return 0.0
        
        time_key = calculate_time_weight(year, self.current_year)
        time_weight = TIME_WEIGHTS.get(time_key, 0.0)
        
        # L4: 권리상태 가중치
        metadata = doc.get("metadata", {})
        status_value = metadata.get("kipris_register_status", "")
        status_key = map_patent_status(status_value)
        status_weight = PATENT_STATUS_WEIGHTS.get(status_key, 0.0)
        
        # 최종 점수 = 시간 × 권리상태
        score = time_weight * status_weight
        
        return score
    
    def _calculate_project_score(self, doc: Dict) -> float:
        """
        연구과제 문서의 AHP 점수 계산
        L1(시간) × L3(규모)
        
        Args:
            doc: 연구과제 문서 데이터
            
        Returns:
            점수 (0~1)
        """
        # L1: 시간 가중치
        year = doc.get("year")
        if not year:
            return 0.0
        
        time_key = calculate_time_weight(year, self.current_year)
        time_weight = TIME_WEIGHTS.get(time_key, 0.0)
        
        # L3: 규모 가중치 (연구비)
        metadata = doc.get("metadata", {})
        budget = metadata.get("TOT_RND_AMT", 0.0)
        if isinstance(budget, (int, float)):
            budget_key = map_project_budget(float(budget))
            scale_weight = PROJECT_SCALE_WEIGHTS.get(budget_key, 0.0)
        else:
            scale_weight = 0.0
        
        # 최종 점수 = 시간 × 규모
        score = time_weight * scale_weight
        
        return score
