#!/bin/bash
# =============================================================
# INDigO 파이프라인 종료 후 점검 스크립트
# 분기별 crontab 파이프라인(collect → filter → extract → embed)
# 실행이 끝난 뒤, 정상적으로 완료됐는지 5가지 항목을 확인합니다.
# 1, 3, 4번 항목은 patent / article / project 문서 타입별로 세분화하여 표시합니다.
#
# 사용법:
#   bash check_pipeline_health.sh
# =============================================================

echo "=============================================="
echo " INDigO 파이프라인 점검 시작: $(date)"
echo "=============================================="
echo ""

PASS=0
FAIL=0

DOC_TYPES=("patent" "article" "project")

check_pass() {
    echo "  [정상] $1"
    PASS=$((PASS+1))
}

check_fail() {
    echo "  [확인 필요] $1"
    FAIL=$((FAIL+1))
}

# -------------------------------------------------------------
# 1. 파이프라인이 에러 없이 끝까지 돌았는지 (문서 타입별)
# -------------------------------------------------------------
echo "[1] 파이프라인 종료 상태 확인"

LOG_DIR="/app/data/logs/pipeline"
LATEST_LOG=$(ls -t "$LOG_DIR"/pipeline_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    check_fail "파이프라인 로그 파일을 찾을 수 없습니다. (${LOG_DIR} 확인 필요)"
else
    echo "  최근 로그 파일: $LATEST_LOG"
    if grep -q "Pipeline finished" "$LATEST_LOG"; then
        check_pass "파이프라인 전체가 정상적으로 끝까지 실행되었습니다."
    else
        check_fail "로그에서 'Pipeline finished' 메시지를 찾지 못했습니다. 중간에 중단됐을 가능성이 있습니다."
    fi

    echo "  ── 문서 타입별 에러/실패 로그 확인 ──"
    for doc_type in "${DOC_TYPES[@]}"; do
        TYPE_ERRORS=$(grep -i "^${doc_type}:.*\(failed\|error\)" "$LATEST_LOG")
        if [ -z "$TYPE_ERRORS" ]; then
            check_pass "[${doc_type}] 에러/실패 로그 없음"
        else
            check_fail "[${doc_type}] 에러 로그 발견:"
            echo "$TYPE_ERRORS" | sed 's/^/      /'
        fi
    done

    # 실제 에러만 추출: "ERROR" 로그 레벨, Traceback, Exception 등 진짜 에러 신호만 확인.
    # '"errors": 0', '"failed_docs": 0'처럼 값이 0인 정상 통계(JSON dump)는 제외한다.
    OTHER_ERRORS=$(grep -E "ERROR|CRITICAL|Traceback|Exception" "$LATEST_LOG" \
        | grep -vE '"(errors|failed_docs)":[[:space:]]*0\b' \
        | grep -vE "^(patent|article|project):")
    if [ -n "$OTHER_ERRORS" ]; then
        check_fail "문서 타입과 무관한 일반 에러 로그가 발견됐습니다:"
        echo "$OTHER_ERRORS" | sed 's/^/      /'
    else
        check_pass "문서 타입과 무관한 일반 에러 로그 없음"
    fi
fi
echo ""

# -------------------------------------------------------------
# 2. 검색 서비스(API)가 정상적으로 재시작됐는지
# -------------------------------------------------------------
echo "[2] 검색 서비스(API) 재시작 확인"

if [ -n "$LATEST_LOG" ]; then
    if grep -q "API container stopped" "$LATEST_LOG" && grep -q "API container restart" "$LATEST_LOG"; then
        check_pass "API 컨테이너가 정상적으로 중지/재시작 되었습니다."
    else
        check_fail "로그에서 API 컨테이너의 중지 또는 재시작 기록을 찾지 못했습니다."
    fi
fi

if docker ps --format '{{.Names}}' | grep -q "^search_api$"; then
    check_pass "search_api 컨테이너가 현재 정상적으로 실행 중입니다."
else
    check_fail "search_api 컨테이너가 현재 실행 중이지 않습니다. 'docker ps -a'로 상태를 확인하세요."
fi
echo ""

# -------------------------------------------------------------
# 3. 체크포인트 파일이 정상적으로 정리됐는지 (문서 타입별, 오염 방지 확인)
# -------------------------------------------------------------
echo "[3] 체크포인트 정리 상태 확인"

CHECKPOINT_DIR="/app/data/checkpoints"

for doc_type in "${DOC_TYPES[@]}"; do
    CHECKPOINT_FILE="${CHECKPOINT_DIR}/extraction_${doc_type}_checkpoint.json"
    if [ -f "$CHECKPOINT_FILE" ]; then
        check_fail "[${doc_type}] 삭제되지 않은 체크포인트 파일이 남아있습니다: ${CHECKPOINT_FILE}"
        echo "      → 다음 실행 시 오염 위험이 있으니 확인 후 필요 시 수동 삭제하세요."
    else
        check_pass "[${doc_type}] 임시 체크포인트 파일이 정상적으로 삭제되었습니다."
    fi
done
echo ""

# -------------------------------------------------------------
# 4. 기대 증가량(필터링 파일) vs 실제 증가량(ChromaDB) 비교 (문서 타입별)
# -------------------------------------------------------------
echo "[4] 기대 증가량 대비 실제 적재 증가량 검증 (문서 타입별)"

HISTORY_FILE="/app/data/logs/chroma_counts_history.json"

CHROMA_CHECK_OUTPUT=$(python3 -c "
import json
from pathlib import Path
from indigo_pipeline.stores.vector_store import ChromaVectorStore
from indigo_pipeline import config

DOC_TYPES = ['patent', 'article', 'project']
ITEM_TYPES = ['chunks', 'entities', 'relations']
HISTORY_FILE = Path('$HISTORY_FILE')

# 1) 이전 실행 시점 기록 불러오기 (문서 타입별 chunks/entities/relations + 필터링 파일 건수)
previous = {}
if HISTORY_FILE.exists():
    try:
        previous = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
    except Exception:
        previous = {}

store = ChromaVectorStore()
current = {}
overall_ok = True

for doc_type in DOC_TYPES:
    print(f'  ── {doc_type} ──')
    current[doc_type] = {}

    # 현재 ChromaDB 건수
    for item_type in ITEM_TYPES:
        name = f'{doc_type}_{item_type}'
        collection = store.collections.get(name)
        current[doc_type][item_type] = collection.count() if collection is not None else 0

    # 현재 필터링 최종 파일의 처리 대상(eligible) 건수
    # 파이프라인의 실제 처리 기준(text 100자 이상)과 동일한 규칙으로 계산한다.
    # 단순히 파일 전체 건수를 쓰면, project처럼 전체 건수는 그대로여도
    # eligible로 분류되는 문서 구성 자체가 바뀌는 경우를 놓친다.
    train_file = Path(config.TRAIN_FILES[doc_type])
    if train_file.exists():
        try:
            records = json.loads(train_file.read_text(encoding='utf-8'))
            if isinstance(records, list):
                total_count = len(records)
                eligible_count = sum(
                    1 for r in records
                    if isinstance(r.get('text'), str) and len(r.get('text')) >= 100
                )
            else:
                total_count = None
                eligible_count = None
        except Exception:
            total_count = None
            eligible_count = None
    else:
        total_count = None
        eligible_count = None
    current[doc_type]['filtered_file_total'] = total_count
    current[doc_type]['filtered_file_eligible'] = eligible_count

    prev = previous.get(doc_type, {})
    prev_chunks = prev.get('chunks')
    prev_eligible = prev.get('filtered_file_eligible')
    curr_chunks = current[doc_type]['chunks']

    if prev_chunks is None or prev_eligible is None or eligible_count is None:
        print(f'    이전 기록이 없어 증가량 비교를 건너뜁니다. (이번 실행 결과는 다음 비교를 위해 저장됩니다)')
        continue

    expected_increase = eligible_count - prev_eligible   # 처리 대상(eligible) 문서 수 기준 기대 증가량
    actual_increase = curr_chunks - prev_chunks           # ChromaDB 기준, 실제로 늘어난 문서 수
    excluded_amount = expected_increase - actual_increase  # 기대했지만 실제로 반영되지 않은 양 (음수면 초과 적재)

    print(f'    증가 전 양(이전 chunks)      : {prev_chunks}건')
    print(f'    증가 후 양(현재 chunks)      : {curr_chunks}건')
    print(f'    증가해야 하는 양(eligible 기준): {expected_increase}건  (필터링 전체 {total_count}건 중 처리 대상 {eligible_count}건)')
    print(f'    실제 증가한 양(ChromaDB 기준) : {actual_increase}건')
    print(f'    제외된 양(기대 - 실제)        : {excluded_amount}건')

    if expected_increase <= 0:
        if actual_increase > 0:
            print(f'    → eligible 기준 총량은 늘지 않았지만 실제로는 {actual_increase}건이 새로 저장됨.')
            print(f'      (필터링 전체 건수는 유지된 채, 처리 대상으로 분류되는 문서 구성 자체가 바뀌었을 가능성 — 데이터 소스 안정성 참고용으로 기록)')
        else:
            print(f'    → 새로 추가된 필터링 문서가 없어 비교 대상 아님')
        continue

    if actual_increase < 0:
        print(f'    [경고] ChromaDB 건수가 오히려 줄어들었습니다. 데이터 손실 가능성이 있으니 확인하세요.')
        overall_ok = False
    elif excluded_amount < 0:
        print(f'    [경고] 실제 증가량이 기대 증가량보다 많습니다({-excluded_amount}건 초과). 중복 적재 여부를 확인하세요.')
        overall_ok = False
    else:
        excluded_ratio = (excluded_amount / expected_increase * 100) if expected_increase > 0 else 0
        if excluded_ratio > 20:
            print(f'    [경고] 제외된 양의 비율이 {excluded_ratio:.1f}%로 높습니다. 추출 실패 문서가 있는지 확인하세요.')
            overall_ok = False
        else:
            print(f'    → 정상 범위 (제외 비율 {excluded_ratio:.1f}% — 짧은 초록/제목 제외 케이스 등 정상 제외분 포함 가능)')

# 이번 실행 결과를 다음 비교를 위해 저장
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
HISTORY_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')

import sys
sys.exit(0 if overall_ok else 2)
" 2>/tmp/chroma_check_err.log)

CHROMA_EXIT=$?
echo "$CHROMA_CHECK_OUTPUT"

if [ $CHROMA_EXIT -eq 0 ]; then
    check_pass "기대 증가량 대비 실제 적재 증가량이 정상 범위입니다."
elif [ $CHROMA_EXIT -eq 2 ]; then
    check_fail "기대 증가량과 실제 적재 증가량 사이에 경고가 발견되었습니다. 위 [경고] 항목을 확인하세요."
else
    check_fail "ChromaDB 조회에 실패했습니다. 컨테이너 안에서 이 스크립트를 실행 중인지 확인하세요."
    cat /tmp/chroma_check_err.log 2>/dev/null | sed 's/^/    /'
fi
echo ""

# -------------------------------------------------------------
# 5. GPU가 정상적으로 해제됐는지
# -------------------------------------------------------------
echo "[5] GPU 상태 확인"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  현재 GPU 메모리 사용 현황:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
    check_pass "nvidia-smi 조회 완료. 위 사용량이 비정상적으로 높지 않은지 눈으로 확인하세요."
else
    check_fail "nvidia-smi 명령어를 찾을 수 없습니다. GPU 드라이버 환경을 확인하세요."
fi
echo ""

# -------------------------------------------------------------
# 결과 요약
# -------------------------------------------------------------
echo "=============================================="
echo " 점검 결과 요약: 정상 ${PASS}건 / 확인 필요 ${FAIL}건"
echo "=============================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "'확인 필요' 항목이 있습니다. 위 내용을 참고하여 조치해 주세요."
    exit 1
else
    echo ""
    echo "모든 항목이 정상입니다."
    exit 0
fi