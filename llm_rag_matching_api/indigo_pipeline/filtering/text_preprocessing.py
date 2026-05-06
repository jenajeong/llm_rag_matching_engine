"""
텍스트 전처리 공통 함수
수식/기호 제거 및 최소 글자수 필터링
"""

import re
from typing import Optional, Tuple, Any


def remove_formulas_and_symbols(text: str) -> str:
    """
    텍스트에서 수식과 기호 문자를 제거합니다.
    한글, 영문, 숫자, 기본 구두점, 공백만 남깁니다.
    
    제거 대상:
    - 수학 기호 (∑, ∫, √, ∞, ≤, ≥, ≠, ±, ×, ÷ 등)
    - 그리스 문자 (α, β, γ, δ, θ, λ, μ, π, σ, φ, ω 등)
    - 화학식 기호 (→, ↔, ⇌ 등)
    - 특수 기호 (•, ·, ※, ◎, ○, ●, ■, ▲, ▼ 등)
    - LaTeX 수식 패턴 ($...$, $$...$$, \[...\], \(...\))
    - HTML 태그
    - 기타 모든 특수 문자
    
    허용 문자:
    - 한글: 가-힣, ㄱ-ㅎ, ㅏ-ㅣ
    - 영문: a-z, A-Z
    - 숫자: 0-9
    - 기본 구두점: . , ! ? ; : ( ) [ ] { } " ' - 
    - 공백
    
    Args:
        text: 전처리할 텍스트
        
    Returns:
        전처리된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 문자열로 변환
    text = str(text).strip()
    
    if not text:
        return ""
    
    # LaTeX 수식 패턴 제거 ($...$, $$...$$, \[...\], \(...\))
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)  # noqa: W605
    text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)  # noqa: W605
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 한글, 영문, 숫자, 기본 구두점, 공백만 남기고 나머지는 공백으로 치환
    # 허용 문자: 한글(가-힣ㄱ-ㅎㅏ-ㅣ), 영문(a-zA-Z), 숫자(0-9), 공백(\s), 기본 구두점
    # 그리스 문자(δ, λ, σ 등) 및 모든 특수 기호 제거
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,!?;:()\[\]{}"\'-]', ' ', text)
    
    # 연속된 공백을 하나로 통합
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def preprocess_text(text: Any, min_length: int = 100, max_length: int = 5000) -> Tuple[Optional[str], bool]:
    """
    텍스트를 전처리하고 최소/최대 길이를 확인합니다.
    
    Args:
        text: 전처리할 텍스트 (문자열, None, 또는 다른 타입)
        min_length: 최소 글자수 (기본값: 100)
        max_length: 최대 글자수 (기본값: 5000)
        
    Returns:
        (전처리된 텍스트, 길이 조건 만족 여부)
        조건을 만족하지 않으면 (None, False) 반환
    """
    if text is None:
        return None, False
    
    # 문자열로 변환
    if not isinstance(text, str):
        text = str(text)
    
    # 수식 및 기호 제거
    cleaned_text = remove_formulas_and_symbols(text)
    
    # 최소 길이 체크
    if len(cleaned_text) < min_length:
        return None, False
    
    # 최대 길이 체크
    if len(cleaned_text) > max_length:
        return None, False
    
    return cleaned_text, True
