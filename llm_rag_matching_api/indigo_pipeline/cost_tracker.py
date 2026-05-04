import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config


PRICING_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


def _usage_value(response: Any, name: str, default: int = 0) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return default
    return int(getattr(usage, name, default) or default)


class CostTracker:
    def __init__(self, history_file: str | Path | None = None):
        self.history_file = Path(history_file or (config.DATA_DIR / "cost_history.json"))
        self._lock = threading.Lock()
        self.current_task: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def start_task(self, task_type: str, description: str = "") -> None:
        self.current_task = {
            "task_type": task_type,
            "description": description,
            "started_at": datetime.now().isoformat(),
            "events": [],
            "total_cost_usd": 0.0,
        }

    def log_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            if self.current_task is not None:
                self.current_task["events"].append(event)
                self.current_task["total_cost_usd"] += float(event.get("cost_usd", 0.0) or 0.0)

    def get_current_task_summary(self) -> dict[str, Any]:
        if self.current_task is None:
            return {}
        return {
            "task_type": self.current_task["task_type"],
            "description": self.current_task["description"],
            "started_at": self.current_task["started_at"],
            "event_count": len(self.current_task["events"]),
            "total_cost_usd": self.current_task["total_cost_usd"],
        }

    def end_task(self, **metadata) -> dict[str, Any] | None:
        if self.current_task is None:
            return None
        result = self.get_current_task_summary()
        result["ended_at"] = datetime.now().isoformat()
        result["metadata"] = metadata
        self._append_history(result)
        self.current_task = None
        return result

    def _append_history(self, result: dict[str, Any]) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if self.history_file.exists():
            try:
                history = json.loads(self.history_file.read_text(encoding="utf-8"))
            except Exception:
                history = []
        if not isinstance(history, list):
            history = []
        history.append(result)
        self.history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_history(self) -> None:
        if self.current_task is not None:
            self._append_history(self.get_current_task_summary())


_TRACKER = CostTracker()


def get_cost_tracker() -> CostTracker:
    return _TRACKER


def log_chat_usage(component: str, model: str, response: Any) -> None:
    prompt_tokens = _usage_value(response, "prompt_tokens")
    completion_tokens = _usage_value(response, "completion_tokens")
    pricing = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])
    _TRACKER.log_event({
        "component": component,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": _usage_value(response, "total_tokens", prompt_tokens + completion_tokens),
        "cost_usd": cost,
        "created_at": datetime.now().isoformat(),
    })


def log_embedding_usage(component: str, model: str, response: Any) -> None:
    prompt_tokens = _usage_value(response, "prompt_tokens", _usage_value(response, "total_tokens"))
    pricing = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    _TRACKER.log_event({
        "component": component,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": _usage_value(response, "total_tokens", prompt_tokens),
        "cost_usd": prompt_tokens / 1000 * pricing["input"],
        "created_at": datetime.now().isoformat(),
    })
