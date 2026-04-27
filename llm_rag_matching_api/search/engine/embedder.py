"""
Embedder
임베딩 모델 구현 - Qwen3 (GPU) / OpenAI (API) 자동 전환
"""

from typing import List, Union

import numpy as np

from .cost_tracker import log_embedding_usage
from .settings import (
    QWEN_EMBEDDING_MODEL, QWEN_EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM,
    OPENAI_API_KEY
)


class Embedder:
    """임베딩 모델 클래스 - GPU/API 자동 전환 (싱글톤)"""

    _instance = None
    _initialized = False

    def __new__(cls, force_api: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, force_api: bool = False):
        """
        임베딩 모델 초기화

        Args:
            force_api: True면 GPU 유무와 관계없이 OpenAI API 사용
        """
        # 이미 초기화되었으면 스킵
        if Embedder._initialized:
            return

        self.force_api = force_api
        self.model = None
        self.tokenizer = None
        self.client = None
        self.use_gpu = False

        self._init_model()
        Embedder._initialized = True

    def _init_model(self):
        """환경에 맞는 모델 초기화"""
        if self.force_api:
            print("Using OpenAI API (forced)")
            self._init_openai()
            return

        # CUDA 체크
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                self._init_qwen()
                self.use_gpu = True
            else:
                print("CUDA not available, falling back to OpenAI API")
                self._init_openai()
        except ImportError:
            print("PyTorch not installed, using OpenAI API")
            self._init_openai()

    def _init_qwen(self):
        """Qwen3 모델 로드 (GPU)"""
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading Qwen3 model: {QWEN_EMBEDDING_MODEL}")
        print("This may take a while on first run (downloading ~16GB)...")

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(
            QWEN_EMBEDDING_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print("Qwen3 model loaded successfully")

    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"OpenAI client initialized with model: {OPENAI_EMBEDDING_MODEL}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 텍스트 리스트 또는 단일 텍스트

        Returns:
            임베딩 벡터 (numpy array)
            - 단일 텍스트: 1차원 (dim,)
            - 리스트: 2차원 (n, dim)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if self.use_gpu:
            embeddings = self._encode_qwen(texts)
        else:
            embeddings = self._encode_openai(texts)

        if single_input:
            return embeddings[0]
        return embeddings

    def _encode_qwen(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Qwen3로 임베딩 생성 (배치 처리로 메모리 관리)"""
        import torch

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(self.model.device)

                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())

            # 메모리 정리
            del inputs, outputs, embeddings
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """OpenAI API로 임베딩 생성"""
        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts
        )

        # 비용 추적
        log_embedding_usage(
            component="embedding",
            model=OPENAI_EMBEDDING_MODEL,
            response=response
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        """임베딩 차원 반환"""
        if self.use_gpu:
            return QWEN_EMBEDDING_DIM
        else:
            return OPENAI_EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        if self.use_gpu:
            return QWEN_EMBEDDING_MODEL
        else:
            return OPENAI_EMBEDDING_MODEL


if __name__ == "__main__":
    # 테스트
    embedder = Embedder()
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # 샘플 텍스트 임베딩
    test_texts = ["딥러닝 기반 의료영상 분석", "자연어 처리 연구"]
    embeddings = embedder.encode(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
