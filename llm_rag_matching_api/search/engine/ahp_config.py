"""
AHP 설정
AHP 가중치 및 문서 유형 순위 설정
(로컬 전용: 이 파일은 .gitignore에 있음)
"""

# ============================================================
# 1단계: 문서 유형 가중치
# ============================================================
DEFAULT_TYPE_WEIGHTS = {
    "article": 0.4,     # 논문
    "patent": 0.2,      # 특허
    "project": 0.4      # 연구과제
}

# ============================================================
# 2단계: L1 - 시간 (모든 문서 공통)
# ============================================================
TIME_WEIGHTS = {
    "0-2": 0.56,    # 0~2년
    "3-5": 0.26,    # 3~5년
    "6-8": 0.12,    # 6~8년
    "9-11": 0.06    # 9~11년
}

# ============================================================
# 2단계: L2 - 기여도 (논문용)
# ============================================================
ARTICLE_CONTRIBUTION_WEIGHTS = {
    "제1저자": 0.52,
    "공동_교신": 0.20,
    "공동_제1저자": 0.20,
    "공동_참여": 0.08
}

# ============================================================
# 2단계: L3 - 규모
# ============================================================

# 논문 규모 가중치 (학술지 등급)
ARTICLE_SCALE_WEIGHTS = {
    "학술지등급(SCI등)": 0.52,
    "학술_국내_등급": 0.20,
    "학술_국제_등급": 0.20,
    "기타_등급": 0.08
}

# 연구과제 규모 가중치 (연구비 규모)
PROJECT_SCALE_WEIGHTS = {
    "5억_이상": 0.56,
    "3억_이상_5억_미만": 0.26,
    "5천만이상_3억_미만": 0.12,
    "5천만미만": 0.06
}

# 특허는 L3 규모 계층 없음

# ============================================================
# 2단계: L4 - 권리상태 (특허용)
# ============================================================
PATENT_STATUS_WEIGHTS = {
    "등록": 0.75,
    "출원": 0.25
}

# ============================================================
# 유형별 문서 1건 최대 점수 (정규화용)
# 논문: L1×L2×L3, 특허: L1×L4, 연구과제: L1×L3 이론상 최대
# ============================================================
MAX_ARTICLE_SCORE_PER_DOC = max(TIME_WEIGHTS.values()) * max(ARTICLE_CONTRIBUTION_WEIGHTS.values()) * max(ARTICLE_SCALE_WEIGHTS.values())
MAX_PATENT_SCORE_PER_DOC = max(TIME_WEIGHTS.values()) * max(PATENT_STATUS_WEIGHTS.values())
MAX_PROJECT_SCORE_PER_DOC = max(TIME_WEIGHTS.values()) * max(PROJECT_SCALE_WEIGHTS.values())

# ============================================================
# 메타데이터 필드 매핑 (data/train 디렉토리 문서 참조)
# ============================================================

# 논문 (Article) 메타데이터 필드
ARTICLE_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 가중치)
    "contribution": "metadata.THSS_PATICP_GBN",       # 기여도 (L2 가중치)
    "journal_type": "metadata.JRNL_GBN"                # 학술지 등급 (L3 규모 가중치)
}

# 특허 (Patent) 메타데이터 필드
PATENT_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 가중치)
    "status": "metadata.kipris_register_status"       # 권리상태 (L4 가중치)
}

# 연구과제 (Project) 메타데이터 필드
PROJECT_METADATA_FIELDS = {
    "year": "year",                                    # 연도 (L1 시간 가중치)
    "budget": "metadata.TOT_RND_AMT"                   # 연구비 규모 (L3 규모 가중치)
}

# ============================================================
# 각 문서 유형별 기준 가중치 정의
# ============================================================

# 논문 (Article) 기준 가중치
# L1(시간) + L2(기여도) + L3(규모)
ARTICLE_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "contribution": 1.0,      # L2 기여도 가중치 (ARTICLE_CONTRIBUTION_WEIGHTS 사용)
    "scale": 1.0              # L3 규모 가중치 (ARTICLE_SCALE_WEIGHTS 사용)
}

# 특허 (Patent) 기준 가중치
# L1(시간) + L4(권리상태)
PATENT_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "status": 1.0             # L4 권리상태 가중치 (PATENT_STATUS_WEIGHTS 사용)
}

# 연구과제 (Project) 기준 가중치
# L1(시간) + L3(규모)
PROJECT_CRITERIA_WEIGHTS = {
    "time": 1.0,              # L1 시간 가중치 (TIME_WEIGHTS 사용)
    "scale": 1.0              # L3 규모 가중치 (PROJECT_SCALE_WEIGHTS 사용)
}

# ============================================================
# 메타데이터 값 → 가중치 키 매핑 함수
# ============================================================

def map_article_contribution(value: str) -> str:
    """
    논문 기여도 값을 AHP 가중치 키로 매핑

    Args:
        value: metadata.THSS_PATICP_GBN 값
            - "제1저자"
            - "공동(제1)"
            - "공동(교신)"
            - "공동(참여)"

    Returns:
        AHP 가중치 키
    """
    mapping = {
        "제1저자": "제1저자",
        "공동(제1)": "공동_제1저자",
        "공동(교신)": "공동_교신",
        "공동(참여)": "공동_참여"
    }
    return mapping.get(value, "공동_참여")  # 기본값: 최하위 가중치


def map_article_journal_type(value: str) -> str:
    """
    논문 학술지 등급을 AHP 가중치 키로 매핑

    Args:
        value: metadata.JRNL_GBN 값
            - "학술지등급(SCI등)"
            - "학술 국내 등급"
            - "학술국제등급(학술지목록[국제])" -> "학술_국제_등급"
            - "학술 국내 등급" -> "학술_국내_등급"

    Returns:
        AHP 가중치 키
    """
    if not value:
        return "기타_등급"
    if "학술지등급(SCI등)" in value or "SCI" in value:
        return "학술지등급(SCI등)"
    elif "학술" in value and "국내" in value:
        return "학술_국내_등급"
    elif "학술국제등급" in value or "학술 국제" in value:
        return "학술_국제_등급"
    elif "학술" in value and "국내" in value:
        return "학술_국내_등급"
    else:
        return "기타_등급"


def map_patent_status(value: str) -> str:
    """
    특허 권리상태를 AHP 가중치 키로 매핑

    Args:
        value: metadata.kipris_register_status 값
            - "등록"
            - "출원"

    Returns:
        AHP 가중치 키
    """
    mapping = {
        "등록": "등록",
        "출원": "출원"
    }
    return mapping.get(value, "출원")  # 기본값: 최하위 가중치


def map_project_budget(amount: float) -> str:
    """
    연구과제 연구비를 AHP 가중치 키로 매핑

    Args:
        amount: metadata.TOT_RND_AMT 값 (원 단위)

    Returns:
        AHP 가중치 키
    """
    if amount is None or amount <= 0:
        return "5천만미만"
    if amount > 500000000:  # 5억 이상
        return "5억_이상"
    elif amount > 300000000:  # 3억 이상 ~ 5억 미만
        return "3억_이상_5억_미만"
    elif amount > 50000000:  # 5천만 이상 ~ 3억 미만
        return "5천만이상_3억_미만"
    else:  # 5천만 미만
        return "5천만미만"


def calculate_time_weight(year: int, current_year: int = None) -> str:
    """
    연도를 기준으로 시간 가중치 키 계산

    Args:
        year: 논문 연도
        current_year: 현재 연도 (None이면 datetime.now().year 사용)

    Returns:
        AHP 가중치 키 ("0-2", "3-5", "6-8", "9-11")
    """
    if current_year is None:
        from datetime import datetime
        current_year = datetime.now().year

    age = current_year - year

    if age <= 2:
        return "0-2"
    elif age <= 5:
        return "3-5"
    elif age <= 8:
        return "6-8"
    elif age <= 11:
        return "9-11"
    else:
        return "9-11"  # 11년 이상도 동일 가중치


# ============================================================
# 일관성 검사 임계값
# ============================================================
CONSISTENCY_THRESHOLD = 0.1  # CR < 0.1이면 일관성 만족
