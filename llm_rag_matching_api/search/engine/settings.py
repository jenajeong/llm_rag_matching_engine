import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_RESULTS_DIR = RESULTS_DIR / "runs" / "report"

ARTICLE_DATA_FILE = DATA_DIR / "article" / "article.json"
PATENT_DATA_FILE = DATA_DIR / "patent" / "patent.json"
PROJECT_DATA_FILE = DATA_DIR / "project" / "project.json"

TRAIN_DATA_DIR = DATA_DIR / "train"
DATA_TRAIN_ARTICLE_FILE = TRAIN_DATA_DIR / "article_filtering.json"
DATA_TRAIN_PATENT_FILE = TRAIN_DATA_DIR / "patent_filtering.json"
DATA_TRAIN_PROJECT_FILE = TRAIN_DATA_DIR / "project_filtering.json"

RAG_STORE_DIR = DATA_DIR / "rag_store_v2"

QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_EMBEDDING_DIM = 4096
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"
REPORT_SUMMARY_MAX_CHARS = int(os.environ.get("REPORT_SUMMARY_MAX_CHARS", "500"))
REPORT_MAX_TOKENS = int(os.environ.get("REPORT_MAX_TOKENS", "4096"))

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.70
RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 5

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
