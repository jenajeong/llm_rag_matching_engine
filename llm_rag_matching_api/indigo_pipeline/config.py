import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DJANGO_PROJECT_DIR = PACKAGE_DIR.parent
REPO_DIR = DJANGO_PROJECT_DIR.parent
DEFAULT_ENV_FILE = DJANGO_PROJECT_DIR / ".env"


def load_env(env_file: str | Path | None = None) -> None:
    path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


load_env()

DATA_DIR = _path_env("INDIGO_PIPELINE_DATA_DIR", DJANGO_PROJECT_DIR / "data")
TRAIN_DATA_DIR = _path_env("INDIGO_PIPELINE_TRAIN_DATA_DIR", DATA_DIR / "train")
CHECKPOINT_DIR = _path_env("INDIGO_PIPELINE_CHECKPOINT_DIR", DATA_DIR / "checkpoints")
RESULTS_DIR = _path_env("INDIGO_PIPELINE_RESULTS_DIR", DJANGO_PROJECT_DIR / "results")
LOG_DIR = _path_env("INDIGO_PIPELINE_LOG_DIR", DJANGO_PROJECT_DIR / "logs")
RAG_STORE_DIR = _path_env("INDIGO_RAG_STORE_DIR", DATA_DIR / "rag_store_v2")

DATA_TRAIN_ARTICLE_FILE = TRAIN_DATA_DIR / "article_filtering.json"
DATA_TRAIN_PATENT_FILE = TRAIN_DATA_DIR / "patent_filtering.json"
DATA_TRAIN_PROJECT_FILE = TRAIN_DATA_DIR / "project_filtering.json"
TRAIN_FILES = {
    "article": DATA_TRAIN_ARTICLE_FILE,
    "patent": DATA_TRAIN_PATENT_FILE,
    "project": DATA_TRAIN_PROJECT_FILE,
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
QWEN_EMBEDDING_MODEL = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
QWEN_EMBEDDING_DIM = int(os.environ.get("QWEN_EMBEDDING_DIM", "4096"))
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = int(os.environ.get("OPENAI_EMBEDDING_DIM", "1536"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.70"))
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", "5"))
ENTITY_EXTRACTION_MAX_CHARS = int(os.environ.get("INDIGO_ENTITY_EXTRACTION_MAX_CHARS", "8000"))
LLM_MAX_RETRIES = int(os.environ.get("INDIGO_LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT = float(os.environ.get("INDIGO_LLM_REQUEST_TIMEOUT", "120"))
