"""
LightRAG 프롬프트 통합 관리 모듈
- Index Time: 엔티티/관계 추출
- Query Time: 키워드 추출

산학협력 매칭 시스템용 커스텀 프롬프트
- 논문, 특허, R&D 과제에서 일관된 Entity 추출
- Entity Types: Target, Problem, Solution, Achievement
"""

# ============================================================
# 구분자 설정 (LightRAG 공식)
# ============================================================
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"


# ============================================================
# 엔티티 타입 설정
# ============================================================
# 산학협력 매칭에 최적화된 Entity 타입
DEFAULT_ENTITY_TYPES = [
    "target",       # 해결하고자 하는 대상 또는 주체
    "problem",      # 연구대상(Target)의 기존 한계나 부정적 현상
    "solution",     # 문제(Problem) 해결을 위한 구체적 방법/기술/알고리즘
    "achievement",  # 제안된 방법으로 달성한 결과/성과
]


# ============================================================
# Index Time: 엔티티/관계 추출 프롬프트 (산학협력 매칭용)
# ============================================================
ENTITY_EXTRACTION_PROMPT = """
-Goal-
산학협력 매칭 시스템을 위한 엔티티/관계 추출입니다.
- 입력: 논문, 특허, R&D 과제 텍스트
- 목적: 기업 기술 수요 ↔ 교수 연구 역량 매칭
- 핵심: 검색 가능한 구체적 기술 용어 추출 (일반적 용어 ✗)

-Entity Types-
| 타입 | 정의 | 핵심 질문 | 타입 판별 힌트 | 제외 |
|------|------|----------|----------|------|
| target (연구 대상) | 문제 해결 및 연구 성과에서 해결하고자 하는 대상 또는 주체 | 무엇을 연구하는가? | 센서, 배터리| 행위(→solution), 효과(→achievement) |
| problem (문제) | 연구대상(Target)의 기존 한계나 부정적인 현상 | 어떤 문제를 해결하려는가? | "~한계", "~어렵다", "~부족하다" | 배경설명, 일반론 |
| solution (해결책) | 발생한 문제(Problem) 해결을 위한 구체적인 방법, 기술, 알고리즘, 절차 | 어떻게 해결하는가? | 파인튜닝, 표면코팅, Few-shot Learning | 단독 행위(분석, 처리), 상위개념 |
| achievement (성과) | 제안된 방법으로 달성한 결과, 성과 (정량적/정성적) | 무엇을 달성했는가? | "수명 200% 향상", "오차 0.11% 달성" | 맥락 없는 수치, 일반론 |

-Rules-
1. 명시적 추출만: 텍스트에 직접 언급된 내용만, 한국어, 명사/명사구 형태
2. 열거형 분리: 쉼표(,)로 나열된 독립적 항목 → 개별 엔티티로 분리 (단, 하나의 개념: "수도 및 하수도", "연구 및 개발"은 유지)
3. 구체적 표현 우선: 더 구체적인 표현이 있으면 상위 개념 제외
4. 맥락 없는 수치 금지: ✗ "20%", "0.11%" → ✓ "수명 200% 향상"

-BLACKLIST (엔티티 이름에 단독 사용 금지)-
아래 단어는 검색 변별력이 없으므로 엔티티 이름에 **단독으로 사용 금지**:

| 적용 타입 | 금지 단어 |
|----------|----------|
| 공통 (모든 타입) | "기술", "개발", "연구", "분석", "처리", "시스템", "방법" |
| target | "장치", "데이터" |
| problem | "문제", "한계", "부족", "어려움", "이슈" |
| solution | "적용" |
| achievement | "향상", "개선", "최적화", "성능", "효율" |

금지 예시:
- ✗ "기술 개발" → 무엇의 기술인지 불명확
- ✗ "시스템 구축" → 무슨 시스템인지 불명확
- ✗ "성능 향상" → 무엇의 성능인지 불명확

-Guidelines (좋은 엔티티의 특징, 권장사항)-
아래 특성을 가진 엔티티가 검색에 효과적입니다 (강제 아님, 권장):

| 타입 | 좋은 엔티티의 특징 | 좋은 예 (✓) |
|------|-------------------|-------------|
| target | ① 고유명사 포함 (PointNet++, BERT)<br>② 도메인 특화 물질/장치명 (양극재, LiDAR)<br>③ 구체적 수식어 포함 | "리튬이온 배터리 양극재", "LiDAR 포인트 클라우드" |
| problem | ① [구체적 맥락] + [문제 표현] 형태<br>② 원인이 명시된 한계 | "고온 구조적 불안정성", "한국어 특성 반영 부족" |
| solution | ① 알고리즘 고유명사 (Transformer, CNN)<br>② 구체적 기법명 (Few-shot Learning) | "KoBERT 파인튜닝", "표면 코팅 기술" |
| achievement | ① 정량적 수치 포함 ("95% 정확도")<br>② 구체적 지표명 + 방향 | "사이클 수명 200% 향상", "처리 속도 10배 향상" |

-Domain Context (도메인 맥락 필수)-
엔티티 이름과 관계 키워드에는 반드시 **도메인 컨텍스트**를 포함해야 합니다:

| 구분 | 잘못된 예 (✗) | 올바른 예 (✓) | 이유 |
|------|-------------|--------------|------|
| 엔티티 이름 | "탐지도 4.5 X 10^12 JONES" | "포토디텍터의 탐지도 4.5 X 10^12 JONES" | "탐지"가 다른 도메인(이상치 탐지)과 혼동 |
| 엔티티 이름 | "효율 개선" | "전기차 BMS 기반 배터리 효율 개선" | 무엇의 효율인지 명확화 |
| 엔티티 이름 | "도핑 기법" | "배터리 양극재의 도핑 기법" | 반도체 도핑과 구분 |
| 관계 키워드 | "결함 탐지, 정확도 향상" | "콘크리트 균열 탐지, NDT 정확도 향상" | 도메인별 구분 필요 |

핵심 원칙:
- **solution**: "[도메인]용 [기법]" 또는 "[도메인]의 [기법]" 형태 (예: "배터리 양극재용 표면 코팅 기술")
- **achievement**: "[도메인]의 [지표] [수치/방향]" 형태 (예: "배터리 양극재의 사이클 수명 200% 향상")
- **relation keywords**: "[도메인] + [행위/결과]" 형태 (예: "자율주행 LiDAR 실시간 처리", "콘크리트 비파괴 검사")

-Steps-
1. 엔티티 추출
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 관계 추출
   - relationship_description: 두 엔티티가 관련된 이유
   - relationship_keywords: 관계의 핵심 개념/주제를 요약하는 키워드 2-3개
     * 도메인 맥락이 있으면 반드시 포함 (예: "배터리 수명 연장", "LiDAR 실시간 처리")
     * 범용어 단독 사용 금지 (예: ✗ "성능 향상", "효율 개선" → ✓ "자율주행 메모리 최적화")
   Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>)

3. 완료 시 {completion_delimiter} 출력

######################
-Examples-
######################
Example 1: 모든 엔티티 타입이 추출되는 케이스

Text:
본 연구에서는 리튬이온 배터리의 양극재 열화 현상을 분석하였다. 기존 양극재는 고온에서 구조적 불안정성 문제가 있었다. 이를 해결하기 위해 표면 코팅 기술과 도핑 기법을 적용하였으며, 이를 통해 사이클 수명이 200% 향상되었다.
################
Output:
("entity"{tuple_delimiter}"리튬이온 배터리 양극재"{tuple_delimiter}"target"{tuple_delimiter}"배터리의 핵심 소재로, 연구가 해결하고자 하는 대상이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"problem"{tuple_delimiter}"양극재의 기존 한계를 나타내는 부정적 현상이므로 problem"){record_delimiter}
("entity"{tuple_delimiter}"배터리 양극재용 표면 코팅 기술"{tuple_delimiter}"solution"{tuple_delimiter}"구조적 불안정성 문제를 해결하기 위한 구체적 기법이며, 도메인(배터리 양극재)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"배터리 양극재의 도핑 기법"{tuple_delimiter}"solution"{tuple_delimiter}"양극재 특성 개선을 위한 구체적 방법이며, 도메인(배터리 양극재)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"배터리 양극재의 사이클 수명 200% 향상"{tuple_delimiter}"achievement"{tuple_delimiter}"제안된 방법으로 달성한 정량적 성과이며, 도메인(배터리 양극재)을 명시하여 검색 변별력 확보"){record_delimiter}
("relationship"{tuple_delimiter}"리튬이온 배터리 양극재"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"양극재가 고온에서 구조적 불안정성 문제를 가짐"{tuple_delimiter}"양극재 소재 한계, 고온 열화 현상"){record_delimiter}
("relationship"{tuple_delimiter}"배터리 양극재용 표면 코팅 기술"{tuple_delimiter}"고온 구조적 불안정성"{tuple_delimiter}"표면 코팅으로 구조적 불안정성 문제를 해결함"{tuple_delimiter}"양극재 표면 코팅 안정화, 배터리 열화 방지"){record_delimiter}
("relationship"{tuple_delimiter}"배터리 양극재의 도핑 기법"{tuple_delimiter}"배터리 양극재의 사이클 수명 200% 향상"{tuple_delimiter}"도핑 기법 적용으로 수명이 향상됨"{tuple_delimiter}"양극재 도핑 기반 특성 개선, 배터리 사이클 수명 연장"){record_delimiter}
{completion_delimiter}

######################
Example 2: Problem이 명시되지 않은 케이스

Text:
본 발명은 전기자동차용 배터리 관리 시스템에 관한 것이다. SOC 추정 알고리즘과 셀 밸런싱 기법을 적용하여 배터리 효율을 최적화한다.
################
Output:
("entity"{tuple_delimiter}"전기자동차용 배터리 관리 시스템"{tuple_delimiter}"target"{tuple_delimiter}"발명이 해결하고자 하는 핵심 대상 시스템이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"전기차 BMS용 SOC 추정 알고리즘"{tuple_delimiter}"solution"{tuple_delimiter}"배터리 효율화를 위한 구체적 알고리즘이며, 도메인(전기차 BMS)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"전기차 BMS의 셀 밸런싱 기법"{tuple_delimiter}"solution"{tuple_delimiter}"배터리 셀 균형을 맞추는 구체적 기법이며, 도메인(전기차 BMS)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"전기차 BMS 기반 배터리 효율 개선"{tuple_delimiter}"achievement"{tuple_delimiter}"시스템을 통해 달성하고자 하는 성과이며, 도메인(전기차 BMS)을 명시하여 검색 변별력 확보"){record_delimiter}
("relationship"{tuple_delimiter}"전기차 BMS용 SOC 추정 알고리즘"{tuple_delimiter}"전기자동차용 배터리 관리 시스템"{tuple_delimiter}"SOC 추정 알고리즘이 배터리 관리 시스템에 적용됨"{tuple_delimiter}"전기차 배터리 충전량 추정, BMS 핵심 로직"){record_delimiter}
("relationship"{tuple_delimiter}"전기차 BMS의 셀 밸런싱 기법"{tuple_delimiter}"전기차 BMS 기반 배터리 효율 개선"{tuple_delimiter}"셀 밸런싱으로 배터리 효율을 개선함"{tuple_delimiter}"전기차 BMS 셀 균형화, EV 배터리 수명 연장"){record_delimiter}
{completion_delimiter}

######################
Example 3: Achievement가 명시되지 않은 케이스

Text:
BERT 기반 한국어 감성분석 모델을 개발하였다. 기존 모델은 한국어 특성 반영이 부족한 한계가 있었다. KoBERT를 파인튜닝하고 형태소 분석을 전처리에 추가하였다.
################
Output:
("entity"{tuple_delimiter}"한국어 감성분석 모델"{tuple_delimiter}"target"{tuple_delimiter}"연구가 개발하고자 하는 핵심 대상이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"problem"{tuple_delimiter}"기존 모델의 한계를 나타내는 부정적 현상이므로 problem"){record_delimiter}
("entity"{tuple_delimiter}"한국어 감성분석용 KoBERT 파인튜닝"{tuple_delimiter}"solution"{tuple_delimiter}"한국어 특성 문제를 해결하기 위한 구체적 기법이며, 도메인(한국어 감성분석)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"한국어 감성분석의 형태소 분석 전처리"{tuple_delimiter}"solution"{tuple_delimiter}"한국어 처리를 위한 구체적 전처리 방법이며, 도메인(한국어 감성분석)을 명시하여 검색 변별력 확보"){record_delimiter}
("relationship"{tuple_delimiter}"한국어 감성분석 모델"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"기존 감성분석 모델이 한국어 특성 반영에 한계가 있음"{tuple_delimiter}"BERT 감성분석 한계, 한국어 형태소 처리"){record_delimiter}
("relationship"{tuple_delimiter}"한국어 감성분석용 KoBERT 파인튜닝"{tuple_delimiter}"한국어 특성 반영 부족"{tuple_delimiter}"KoBERT 파인튜닝으로 한국어 특성 반영 문제를 해결함"{tuple_delimiter}"한국어 감성분석 파인튜닝, 한국어 특화 NLP 모델 개선"){record_delimiter}
{completion_delimiter}

######################
Example 4: Problem과 Achievement 둘 다 없는 케이스

Text:
본 연구는 딥러닝 기반 의료영상 분할 기술을 개발한다. U-Net 아키텍처와 어텐션 메커니즘을 결합하여 CT 영상에서 장기를 자동 분할한다.
################
Output:
("entity"{tuple_delimiter}"의료영상 자동 분할"{tuple_delimiter}"target"{tuple_delimiter}"연구가 개발하고자 하는 핵심 대상이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"CT 영상"{tuple_delimiter}"target"{tuple_delimiter}"분할의 대상이 되는 데이터이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"의료영상 분할용 U-Net 아키텍처"{tuple_delimiter}"solution"{tuple_delimiter}"영상 분할을 위한 구체적 딥러닝 구조이며, 도메인(의료영상 분할)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"CT 영상 분할의 어텐션 메커니즘"{tuple_delimiter}"solution"{tuple_delimiter}"성능 향상을 위한 구체적 딥러닝 기법이며, 도메인(CT 영상 분할)을 명시하여 검색 변별력 확보"){record_delimiter}
("relationship"{tuple_delimiter}"의료영상 분할용 U-Net 아키텍처"{tuple_delimiter}"의료영상 자동 분할"{tuple_delimiter}"U-Net이 의료영상 분할의 기반 구조로 사용됨"{tuple_delimiter}"U-Net 기반 의료영상 분할, CT 장기 세그멘테이션"){record_delimiter}
("relationship"{tuple_delimiter}"CT 영상 분할의 어텐션 메커니즘"{tuple_delimiter}"의료영상 분할용 U-Net 아키텍처"{tuple_delimiter}"어텐션 메커니즘이 U-Net과 결합되어 사용됨"{tuple_delimiter}"어텐션 U-Net 의료영상 결합, CT 분할 정확도 향상"){record_delimiter}
{completion_delimiter}

######################
Example 5: 복잡한 관계가 있는 케이스

Text:
자율주행 차량의 LiDAR 포인트 클라우드 처리 시스템을 연구하였다. 기존 처리 방식은 실시간성 확보가 어려운 문제가 있었다. PointNet++ 기반 경량화 모델과 병렬 처리 파이프라인을 개발하여 처리 속도를 10배 향상시키고 메모리 사용량을 50% 절감하였다.
################
Output:
("entity"{tuple_delimiter}"LiDAR 포인트 클라우드 처리 시스템"{tuple_delimiter}"target"{tuple_delimiter}"연구가 해결하고자 하는 핵심 대상 시스템이므로 target"){record_delimiter}
("entity"{tuple_delimiter}"자율주행 LiDAR의 실시간 처리 어려움"{tuple_delimiter}"problem"{tuple_delimiter}"기존 처리 방식의 한계를 나타내는 부정적 현상이며, 도메인(자율주행 LiDAR)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"자율주행 LiDAR용 PointNet++ 경량화 모델"{tuple_delimiter}"solution"{tuple_delimiter}"실시간성 문제를 해결하기 위한 구체적 딥러닝 모델이며, 도메인(자율주행 LiDAR)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"자율주행 LiDAR의 병렬 처리 파이프라인"{tuple_delimiter}"solution"{tuple_delimiter}"속도 향상을 위한 구체적 처리 구조이며, 도메인(자율주행 LiDAR)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"자율주행 LiDAR의 처리 속도 10배 향상"{tuple_delimiter}"achievement"{tuple_delimiter}"제안된 방법으로 달성한 정량적 성과이며, 도메인(자율주행 LiDAR)을 명시하여 검색 변별력 확보"){record_delimiter}
("entity"{tuple_delimiter}"자율주행 시스템의 메모리 사용량 50% 절감"{tuple_delimiter}"achievement"{tuple_delimiter}"제안된 방법으로 달성한 정량적 성과이며, 도메인(자율주행 시스템)을 명시하여 검색 변별력 확보"){record_delimiter}
("relationship"{tuple_delimiter}"LiDAR 포인트 클라우드 처리 시스템"{tuple_delimiter}"자율주행 LiDAR의 실시간 처리 어려움"{tuple_delimiter}"기존 처리 시스템이 실시간 처리에 한계가 있음"{tuple_delimiter}"LiDAR 데이터 처리 한계, 자율주행 실시간성 병목"){record_delimiter}
("relationship"{tuple_delimiter}"자율주행 LiDAR용 PointNet++ 경량화 모델"{tuple_delimiter}"자율주행 LiDAR의 실시간 처리 어려움"{tuple_delimiter}"경량화 모델로 실시간성 문제를 해결함"{tuple_delimiter}"포인트 클라우드 경량화, 자율주행 실시간 LiDAR 처리"){record_delimiter}
("relationship"{tuple_delimiter}"자율주행 LiDAR의 병렬 처리 파이프라인"{tuple_delimiter}"자율주행 LiDAR의 처리 속도 10배 향상"{tuple_delimiter}"병렬 처리로 속도가 10배 향상됨"{tuple_delimiter}"LiDAR 병렬 연산 파이프라인, 자율주행 포인트 클라우드 고속 처리"){record_delimiter}
("relationship"{tuple_delimiter}"자율주행 LiDAR용 PointNet++ 경량화 모델"{tuple_delimiter}"자율주행 시스템의 메모리 사용량 50% 절감"{tuple_delimiter}"경량화 모델로 메모리 사용이 절감됨"{tuple_delimiter}"PointNet++ 자율주행 경량화, LiDAR 메모리 최적화"){record_delimiter}
{completion_delimiter}

######################
Example 6: ❌ Negative Example - 이렇게 하면 안 됩니다

Text:
본 연구에서는 새로운 기술을 개발하여 시스템의 성능을 향상시켰다. 기존 방법은 한계가 있어 분석과 처리에 어려움이 있었다. 연구팀은 개선된 방법을 적용하여 효율을 최적화하였다.
################
잘못된 Output (❌ BLACKLIST 위반):
("entity"{tuple_delimiter}"기술"{tuple_delimiter}"target"{tuple_delimiter}"연구에서 개발한 기술")
("entity"{tuple_delimiter}"시스템"{tuple_delimiter}"target"{tuple_delimiter}"성능을 향상시킨 시스템")
("entity"{tuple_delimiter}"한계"{tuple_delimiter}"problem"{tuple_delimiter}"기존 방법의 한계")
("entity"{tuple_delimiter}"어려움"{tuple_delimiter}"problem"{tuple_delimiter}"분석과 처리의 어려움")
("entity"{tuple_delimiter}"방법"{tuple_delimiter}"solution"{tuple_delimiter}"개선된 방법")
("entity"{tuple_delimiter}"성능 향상"{tuple_delimiter}"achievement"{tuple_delimiter}"시스템의 성능 향상")
("entity"{tuple_delimiter}"효율 최적화"{tuple_delimiter}"achievement"{tuple_delimiter}"달성한 효율 최적화")

이유:
- "기술", "시스템", "방법"은 BLACKLIST 공통 금지 단어
- "한계", "어려움"은 problem 타입 금지 단어
- "성능 향상", "효율 최적화"는 achievement 타입 금지 단어
- 모두 검색 변별력이 없는 범용 표현

올바른 Output (✓):
(엔티티 추출 없음 - 텍스트에 구체적인 기술 용어가 없으므로 추출하지 않음)
{completion_delimiter}

######################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""


# ============================================================
# Query Time: 키워드 추출 프롬프트 (산학협력 매칭용)
# ============================================================
KEYWORD_EXTRACTION_PROMPT = """---Goal---
산학협력 매칭을 위한 검색 키워드 추출입니다.
- 목적: 기업 기술 수요 ↔ 교수 연구 역량 매칭
- 추출된 키워드로 논문/특허/R&D 과제를 검색합니다.
- 한글 키워드와 영어 번역을 모두 추출합니다 (영어 논문 검색용).

---Keywords---
| 타입 | 정의 | 추출 방식 | 형태 | 예시 |
|------|------|----------|------|------|
| low_level | 구체적 기술 용어 | 쿼리 원문에서 그대로 | 명사/명사구 | "CNN", "양극재", "폐 결절" |
| high_level | 쿼리 의도 요약 | 문장/구 형태로 요약 | ~기반 ~분석, ~개선 등 | "의료영상 기반 진단", "배터리 수명 향상" |
| *_en | 영어 번역 | 한글 키워드의 정확한 영어 학술 용어 | 영어 명사/구 | "anomaly detection", "machine learning" |

---Rules---
1. low_level: 쿼리 원문에 있는 기술 용어만 추출 (추론/연상 금지)
2. high_level: 쿼리 의도를 "~기반", "~분석", "~개선", "~검출" 등의 형태로 요약. 쿼리에 특정 도메인·대상(산업, 제품, 공정, 응용 분야 등)이 언급되어 있으면 high_level에 반드시 해당 도메인을 포함하세요.
3. 복합 명사 유지: 의미적으로 연결된 명사는 하나로 ("의료영상", "행동 분석")
4. 1글자 단어 금지: "문", "관", "기", "물" 등 단독 사용 금지 (복합 명사로만 사용)
5. 요청 표현 제외: "교수님", "전문가", "찾아줘", "알려줘", "필요해요", "하고 싶어요"
6. 영어 번역: 각 한글 키워드에 대응하는 정확한 영어 학술 용어로 번역 (순서 동일하게 유지)
7. 오탈자 교정: 쿼리에 명백한 오탈자/띄어쓰기 오류가 있으면 교정된 형태로 키워드 추출. 의도가 불분명하면 원본 유지.

---BLACKLIST (MUST NOT extract)---
아래 단어는 검색 성능을 저하시키므로 절대 추출하지 마세요:
"기술", "개발", "연구", "기술 개발", "연구 개발", "기술개발", "연구개발", "방법", "시스템", "분석", "처리"

위 단어가 쿼리에 있어도 low_level_keywords에 포함하면 안 됩니다.
단, "감성분석", "행동 분석"처럼 도메인 특화 복합어는 허용됩니다.

---Output---
반드시 순수 JSON만 출력하세요. ```json 같은 마크다운 코드 블록으로 감싸지 마세요.
low_level 1-5개, high_level 2-3개, 각각의 영어 번역 포함

---Examples---
Query: "이상치 탐지 및 머신러닝 관련 연구를 찾고 있어요"
Output: {{"low_level_keywords": ["이상치 탐지", "머신러닝"], "low_level_keywords_en": ["anomaly detection", "machine learning"], "high_level_keywords": ["머신러닝 기반 이상치 탐지", "데이터 기반 이상 패턴 분석"], "high_level_keywords_en": ["machine learning based anomaly detection", "data-driven anomaly pattern analysis"]}}

Query: "공장에서 용접 불량이 자꾸 발생하는데 자동으로 검출하고 싶어요"
Output: {{"low_level_keywords": ["용접 불량", "자동 검출"], "low_level_keywords_en": ["welding defect", "automatic detection"], "high_level_keywords": ["용접 결함 자동 검출", "제조 품질 검사"], "high_level_keywords_en": ["automatic welding defect detection", "manufacturing quality inspection"]}}

Query: "CT 영상에서 폐 결절을 자동으로 찾아내고 싶어요"
Output: {{"low_level_keywords": ["CT 영상", "폐 결절"], "low_level_keywords_en": ["CT image", "pulmonary nodule"], "high_level_keywords": ["의료영상 기반 폐 결절 검출", "CAD 기반 진단"], "high_level_keywords_en": ["medical image based pulmonary nodule detection", "CAD-based diagnosis"]}}

Query: "고객 리뷰를 감성분석 하려고 합니다"
Output: {{"low_level_keywords": ["고객 리뷰", "감성분석"], "low_level_keywords_en": ["customer review", "sentiment analysis"], "high_level_keywords": ["텍스트 기반 감성 분석", "자연어처리 기반 리뷰 분석"], "high_level_keywords_en": ["text-based sentiment analysis", "NLP-based review analysis"]}}

Query: "전기차 배터리 충전 시간이 너무 오래 걸려서 단축하고 싶어요"
Output: {{"low_level_keywords": ["전기차 배터리", "충전 시간"], "low_level_keywords_en": ["electric vehicle battery", "charging time"], "high_level_keywords": ["전기차 배터리 충전 효율 향상", "전기차 급속 충전 기술"], "high_level_keywords_en": ["electric vehicle battery charging efficiency improvement", "electric vehicle fast charging technology"]}}

Query: "스마트팜에 IoT 센서 적용해서 작물 생육 모니터링 하려는데요"
Output: {{"low_level_keywords": ["스마트팜", "IoT 센서", "작물 생육 모니터링"], "low_level_keywords_en": ["smart farm", "IoT sensor", "crop growth monitoring"], "high_level_keywords": ["스마트팜 IoT 기반 생육 모니터링", "스마트팜 정밀 농업"], "high_level_keywords_en": ["smart farm IoT-based growth monitoring", "smart farm precision agriculture"]}}

Query: "베어링 기계에서 제품 실패율 방지를 위해서 머신 러닝 기술을 적용하여 비용 절감 효과"
Output: {{"low_level_keywords": ["베어링", "제품 실패율", "머신 러닝"], "low_level_keywords_en": ["bearing", "product failure rate", "machine learning"], "high_level_keywords": ["베어링 제조 머신러닝 기반 불량 예측", "베어링 생산 비용 절감"], "high_level_keywords_en": ["machine learning based bearing manufacturing defect prediction", "bearing production cost reduction"]}}

######################
Query: {query}
Output:
"""


# ============================================================
# Query Time: RAG 응답 생성 프롬프트 (추후 사용)
# ============================================================
RAG_RESPONSE_PROMPT = """---Role---
You are a helpful assistant responding to questions about data in the tables provided.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---
{response_type}

---Data tables---
{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


# ============================================================
# 유틸리티 함수
# ============================================================
def format_entity_extraction_prompt(input_text: str) -> str:
    """
    엔티티 추출 프롬프트 포맷팅

    Args:
        input_text: 입력 텍스트 (논문 초록, 특허 요약, R&D 과제 설명)

    Returns:
        포맷팅된 프롬프트
    """
    return ENTITY_EXTRACTION_PROMPT.format(
        tuple_delimiter=TUPLE_DELIMITER,
        record_delimiter=RECORD_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        input_text=input_text
    )


def format_keyword_extraction_prompt(query: str) -> str:
    """
    키워드 추출 프롬프트 포맷팅

    Args:
        query: 사용자 쿼리

    Returns:
        포맷팅된 프롬프트
    """
    return KEYWORD_EXTRACTION_PROMPT.format(query=query)


if __name__ == "__main__":
    # 테스트
    print("=== Entity Extraction Prompt Test ===")
    test_text = """본 연구에서는 리튬이온 배터리의 양극재 열화 현상을 분석하였다.
    기존 양극재는 고온에서 구조적 불안정성 문제가 있었다.
    이를 해결하기 위해 표면 코팅 기술과 도핑 기법을 적용하였으며,
    이를 통해 사이클 수명이 200% 향상되었다."""
    prompt = format_entity_extraction_prompt(test_text)
    print(prompt)
    print("\n" + "="*50)

    print("\n=== Keyword Extraction Prompt Test ===")
    test_query = "딥러닝을 활용한 의료영상 진단 전문가를 찾아줘"
    prompt = format_keyword_extraction_prompt(test_query)
    print(prompt)
