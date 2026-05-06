"""
데이터 유형별 텍스트 길이 확인 스크립트
논문, 특허, 과제 데이터의 RAG 텍스트 필드 길이를 분석하고 CSV로 저장합니다.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from indigo_pipeline.config import (
    DATA_TRAIN_ARTICLE_FILE,
    DATA_TRAIN_PATENT_FILE,
    DATA_TRAIN_PROJECT_FILE,
    RESULTS_DIR
)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        데이터 리스트
    """
    path = Path(file_path)
    if not path.exists():
        print(f"경고: 파일을 찾을 수 없습니다: {file_path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def calculate_text_length(text: Any) -> int:
    """
    텍스트의 글자 수를 계산합니다.
    
    Args:
        text: 텍스트 (문자열, None, 또는 다른 타입)
        
    Returns:
        글자 수 (None이거나 문자열이 아닌 경우 0)
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        return 0
    return len(text)


def analyze_article_text_length():
    """
    논문 데이터의 text 필드 길이를 분석하고 CSV로 저장합니다.
    """
    print("논문 데이터 분석 중...")
    
    # 데이터 로드
    articles = load_json_data(DATA_TRAIN_ARTICLE_FILE)
    if not articles:
        print("논문 데이터를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(articles)}개의 논문 데이터를 로드했습니다.")
    
    # DataFrame으로 변환
    df = pd.DataFrame(articles)
    
    # text 필드의 글자 수 계산
    df['text_length'] = df['text'].apply(calculate_text_length)
    
    # 텍스트 길이 기준으로 정렬 (긴 것부터)
    df = df.sort_values('text_length', ascending=False)
    
    # CSV로 저장
    output_dir = Path(RESULTS_DIR) / "check"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "article_text_length_check.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"논문 데이터가 저장되었습니다: {output_file}")
    print(f"  - 총 데이터 수: {len(df)}")
    print(f"  - 최소 글자 수: {df['text_length'].min()}")
    print(f"  - 최대 글자 수: {df['text_length'].max()}")
    print(f"  - 평균 글자 수: {df['text_length'].mean():.2f}")
    print(f"  - 중앙값 글자 수: {df['text_length'].median():.2f}")
    print(f"  - 0자 데이터 수: {len(df[df['text_length'] == 0])}")
    print(f"  - 100자 미만 데이터 수: {len(df[df['text_length'] < 100])}")
    print(f"  - 300자 미만 데이터 수: {len(df[df['text_length'] < 300])}")


def analyze_patent_text_length():
    """
    특허 데이터의 text 필드 길이를 분석하고 CSV로 저장합니다.
    """
    print("\n특허 데이터 분석 중...")
    
    # 데이터 로드
    patents = load_json_data(DATA_TRAIN_PATENT_FILE)
    if not patents:
        print("특허 데이터를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(patents)}개의 특허 데이터를 로드했습니다.")
    
    # DataFrame으로 변환
    df = pd.DataFrame(patents)
    
    # text 필드의 글자 수 계산
    df['text_length'] = df['text'].apply(calculate_text_length)
    
    # 텍스트 길이 기준으로 정렬 (긴 것부터)
    df = df.sort_values('text_length', ascending=False)
    
    # CSV로 저장
    output_dir = Path(RESULTS_DIR) / "check"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "patent_text_length_check.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"특허 데이터가 저장되었습니다: {output_file}")
    print(f"  - 총 데이터 수: {len(df)}")
    print(f"  - 최소 글자 수: {df['text_length'].min()}")
    print(f"  - 최대 글자 수: {df['text_length'].max()}")
    print(f"  - 평균 글자 수: {df['text_length'].mean():.2f}")
    print(f"  - 중앙값 글자 수: {df['text_length'].median():.2f}")
    print(f"  - 0자 데이터 수: {len(df[df['text_length'] == 0])}")
    print(f"  - 100자 미만 데이터 수: {len(df[df['text_length'] < 100])}")
    print(f"  - 150자 미만 데이터 수: {len(df[df['text_length'] < 150])}")


def analyze_project_text_length():
    """
    과제 데이터의 text 필드 길이를 분석하고 CSV로 저장합니다.
    """
    print("\n과제 데이터 분석 중...")
    
    # 데이터 로드
    projects = load_json_data(DATA_TRAIN_PROJECT_FILE)
    if not projects:
        print("과제 데이터를 찾을 수 없습니다.")
        return
    
    print(f"총 {len(projects)}개의 과제 데이터를 로드했습니다.")
    
    # DataFrame으로 변환
    df = pd.DataFrame(projects)
    
    # summary 필드의 글자 수 계산
    df['text_length'] = df['text'].apply(calculate_text_length)
    
    # 텍스트 길이 기준으로 정렬 (긴 것부터)
    df = df.sort_values('text_length', ascending=False)
    
    # CSV로 저장
    output_dir = Path(RESULTS_DIR) / "check"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "project_text_length_check.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"과제 데이터가 저장되었습니다: {output_file}")
    print(f"  - 총 데이터 수: {len(df)}")
    print(f"  - 최소 글자 수: {df['text_length'].min()}")
    print(f"  - 최대 글자 수: {df['text_length'].max()}")
    print(f"  - 평균 글자 수: {df['text_length'].mean():.2f}")
    print(f"  - 중앙값 글자 수: {df['text_length'].median():.2f}")
    print(f"  - 0자 데이터 수: {len(df[df['text_length'] == 0])}")
    print(f"  - 200자 미만 데이터 수: {len(df[df['text_length'] < 200])}")
    print(f"  - 400자 미만 데이터 수: {len(df[df['text_length'] < 400])}")


def main():
    """
    메인 함수: 모든 데이터 유형에 대해 텍스트 길이 분석을 수행합니다.
    """
    print("=" * 60)
    print("데이터 유형별 텍스트 길이 분석 시작")
    print("=" * 60)
    
    # 각 데이터 유형별 분석 수행
    analyze_article_text_length()
    analyze_patent_text_length()
    analyze_project_text_length()
    
    print("\n" + "=" * 60)
    print("모든 분석이 완료되었습니다!")
    print(f"결과 파일은 {Path(RESULTS_DIR) / 'check'} 폴더에 저장되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
