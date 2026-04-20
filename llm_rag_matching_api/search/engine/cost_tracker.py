"""
LLM API 비용 추적 모듈

각 단계별 토큰 사용량과 비용을 추적하고 JSON 파일에 누적 저장합니다.
- Index Time: entity_extraction, embedding
- Query Time: keyword_extraction, rag_response, embedding
- Report: report_generation
- Evaluation: noise_rate_eval
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import threading


# ============================================================
# 모델별 가격표 (USD per 1M tokens, 2025년 기준)
# ============================================================
MODEL_PRICING = {
    # Chat Completions
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}


@dataclass
class ComponentUsage:
    """컴포넌트별 사용량 추적"""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost: float):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost


@dataclass
class TaskRecord:
    """작업 단위 기록"""
    timestamp: str
    task: str  # "indexing", "query", "report", "evaluation"
    description: str = ""
    components: Dict[str, Dict] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """
    LLM API 비용 추적기 (싱글톤)

    사용법:
    ```python
    tracker = get_cost_tracker()

    # 작업 시작
    tracker.start_task("indexing", description="50개 문서 인덱싱")

    # API 호출 후 기록
    tracker.log_usage(
        component="entity_extraction",
        model="gpt-4o-mini",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )

    # 작업 종료 및 저장
    tracker.end_task()

    # 요약 출력
    tracker.print_summary()
    ```
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, history_file: Optional[str] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True

        # 기본 저장 경로
        if history_file is None:
            project_root = Path(__file__).parent.parent.parent
            self.history_file = project_root / "cost_history.json"
        else:
            self.history_file = Path(history_file)

        # 현재 세션 데이터
        self._current_task: Optional[TaskRecord] = None
        self._session_usage: Dict[str, ComponentUsage] = {}

        # 히스토리 로드
        self._history = self._load_history()

    def _load_history(self) -> Dict:
        """기존 히스토리 파일 로드"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "created_at": datetime.now().isoformat(),
            "history": [],
            "totals": {
                "all_time_cost_usd": 0.0,
                "by_component": {},
                "by_model": {}
            }
        }

    def _save_history(self):
        """히스토리 파일 저장"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """토큰 수로부터 비용 계산 (USD)"""
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def start_task(self, task: str, description: str = "", **metadata):
        """
        새 작업 시작

        Args:
            task: 작업 유형 ("indexing", "query", "report", "evaluation")
            description: 작업 설명
            **metadata: 추가 메타데이터 (documents=50, queries=10 등)
        """
        self._current_task = TaskRecord(
            timestamp=datetime.now().isoformat(),
            task=task,
            description=description,
            metadata=metadata
        )
        self._session_usage = {}

    def log_usage(
        self,
        component: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0
    ):
        """
        API 호출 사용량 기록

        Args:
            component: 컴포넌트명 (entity_extraction, keyword_extraction 등)
            model: 모델명 (gpt-4o-mini, text-embedding-3-small 등)
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수 (임베딩은 0)
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # 세션 사용량 업데이트
        if component not in self._session_usage:
            self._session_usage[component] = ComponentUsage()
        self._session_usage[component].add(input_tokens, output_tokens, cost)

        # 전역 통계 업데이트
        totals = self._history["totals"]
        totals["all_time_cost_usd"] += cost

        # 컴포넌트별 통계
        if component not in totals["by_component"]:
            totals["by_component"][component] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }
        comp_stats = totals["by_component"][component]
        comp_stats["calls"] += 1
        comp_stats["input_tokens"] += input_tokens
        comp_stats["output_tokens"] += output_tokens
        comp_stats["cost_usd"] += cost

        # 모델별 통계
        if model not in totals["by_model"]:
            totals["by_model"][model] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }
        model_stats = totals["by_model"][model]
        model_stats["calls"] += 1
        model_stats["input_tokens"] += input_tokens
        model_stats["output_tokens"] += output_tokens
        model_stats["cost_usd"] += cost

    def end_task(self, save: bool = True, **extra_metadata) -> Dict:
        """
        현재 작업 종료 및 기록

        Args:
            save: 즉시 파일에 저장할지 여부
            **extra_metadata: 작업 완료 시 추가할 메타데이터 (docs_processed, entities_extracted 등)

        Returns:
            현재 작업의 요약 정보
        """
        if self._current_task is None:
            return {}

        # 작업 완료 시 추가 메타데이터 병합
        if extra_metadata:
            self._current_task.metadata.update(extra_metadata)

        # 세션 사용량을 작업 기록에 추가
        task_cost = 0.0
        for comp_name, usage in self._session_usage.items():
            self._current_task.components[comp_name] = asdict(usage)
            task_cost += usage.cost_usd

        self._current_task.total_cost_usd = task_cost

        # 히스토리에 추가
        task_dict = asdict(self._current_task)
        self._history["history"].append(task_dict)

        if save:
            self._save_history()

        result = task_dict.copy()
        self._current_task = None
        self._session_usage = {}

        return result

    def get_current_task_summary(self) -> Dict:
        """현재 진행 중인 작업의 요약"""
        if self._current_task is None:
            return {"status": "no_active_task"}

        total_cost = sum(u.cost_usd for u in self._session_usage.values())
        total_calls = sum(u.calls for u in self._session_usage.values())

        return {
            "task": self._current_task.task,
            "description": self._current_task.description,
            "total_calls": total_calls,
            "total_cost_usd": total_cost,
            "by_component": {
                name: asdict(usage)
                for name, usage in self._session_usage.items()
            }
        }

    def get_totals(self) -> Dict:
        """전체 누적 통계 반환"""
        return self._history["totals"]

    def print_summary(self):
        """현재 작업 요약 출력"""
        summary = self.get_current_task_summary()

        if summary.get("status") == "no_active_task":
            print("No active task")
            return

        print("\n" + "="*60)
        print(f"Task: {summary['task']} - {summary['description']}")
        print("="*60)

        for comp_name, usage in summary.get("by_component", {}).items():
            print(f"\n{comp_name}:")
            print(f"  Calls: {usage['calls']}")
            print(f"  Input tokens: {usage['input_tokens']:,}")
            print(f"  Output tokens: {usage['output_tokens']:,}")
            print(f"  Cost: ${usage['cost_usd']:.6f}")

        print("\n" + "-"*60)
        print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
        print("="*60 + "\n")

    def print_all_time_summary(self):
        """전체 누적 통계 출력"""
        totals = self._history["totals"]

        print("\n" + "="*60)
        print("ALL-TIME COST SUMMARY")
        print("="*60)

        print(f"\nTotal Cost: ${totals['all_time_cost_usd']:.6f}")

        print("\n--- By Component ---")
        for comp, stats in totals.get("by_component", {}).items():
            print(f"\n{comp}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out")
            print(f"  Cost: ${stats['cost_usd']:.6f}")

        print("\n--- By Model ---")
        for model, stats in totals.get("by_model", {}).items():
            print(f"\n{model}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out")
            print(f"  Cost: ${stats['cost_usd']:.6f}")

        print("\n" + "="*60 + "\n")


# 싱글톤 인스턴스 접근 함수
_tracker_instance: Optional[CostTracker] = None

def get_cost_tracker(history_file: Optional[str] = None) -> CostTracker:
    """CostTracker 싱글톤 인스턴스 반환"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CostTracker(history_file)
    return _tracker_instance


def reset_tracker():
    """테스트용: 트래커 인스턴스 리셋"""
    global _tracker_instance
    _tracker_instance = None
    CostTracker._instance = None


# ============================================================
# 편의 함수: API 응답에서 직접 로깅
# ============================================================
def log_chat_usage(
    component: str,
    model: str,
    response
):
    """
    OpenAI Chat API 응답에서 직접 사용량 로깅

    Args:
        component: 컴포넌트명
        model: 모델명
        response: OpenAI ChatCompletion 응답 객체
    """
    tracker = get_cost_tracker()

    if hasattr(response, 'usage') and response.usage:
        tracker.log_usage(
            component=component,
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )


def log_embedding_usage(
    component: str,
    model: str,
    response
):
    """
    OpenAI Embedding API 응답에서 직접 사용량 로깅

    Args:
        component: 컴포넌트명
        model: 모델명
        response: OpenAI Embedding 응답 객체
    """
    tracker = get_cost_tracker()

    if hasattr(response, 'usage') and response.usage:
        tracker.log_usage(
            component=component,
            model=model,
            input_tokens=response.usage.total_tokens,
            output_tokens=0
        )


if __name__ == "__main__":
    # 테스트
    reset_tracker()
    tracker = get_cost_tracker()

    # 인덱싱 작업 시뮬레이션
    tracker.start_task("indexing", description="테스트 인덱싱 10개 문서", documents=10)

    # 엔티티 추출 호출 시뮬레이션
    for i in range(10):
        tracker.log_usage(
            component="entity_extraction",
            model="gpt-4o-mini",
            input_tokens=2000,
            output_tokens=500
        )

    # 임베딩 호출 시뮬레이션
    tracker.log_usage(
        component="embedding",
        model="text-embedding-3-small",
        input_tokens=50000,
        output_tokens=0
    )

    tracker.print_summary()
    result = tracker.end_task()

    print("\nTask Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 전체 통계
    tracker.print_all_time_summary()
