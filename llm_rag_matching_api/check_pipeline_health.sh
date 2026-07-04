#!/bin/bash
# =============================================================
# INDigO 파이프라인 종료 후 점검 스크립트
# 분기별 crontab 파이프라인(collect → filter → extract → embed)
# 실행이 끝난 뒤, 정상적으로 완료됐는지 5가지 항목을 확인합니다.
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

check_pass() {
    echo "  [정상] $1"
    PASS=$((PASS+1))
}

check_fail() {
    echo "  [확인 필요] $1"
    FAIL=$((FAIL+1))
}

# -------------------------------------------------------------
# 1. 파이프라인이 에러 없이 끝까지 돌았는지
# -------------------------------------------------------------
echo "[1] 파이프라인 종료 상태 확인"

LOG_DIR="/app/data/logs/pipeline"
LATEST_LOG=$(ls -t "$LOG_DIR"/pipeline_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    check_fail "파이프라인 로그 파일을 찾을 수 없습니다. (${LOG_DIR} 확인 필요)"
else
    echo "  최근 로그 파일: $LATEST_LOG"
    if grep -q "Pipeline finished" "$LATEST_LOG"; then
        check_pass "파이프라인이 정상적으로 끝까지 실행되었습니다."
    else
        check_fail "로그에서 'Pipeline finished' 메시지를 찾지 못했습니다. 중간에 중단됐을 가능성이 있습니다."
    fi

    ERROR_LINES=$(grep -i "failed\|error" "$LATEST_LOG")
    if [ -z "$ERROR_LINES" ]; then
        check_pass "로그에 에러/실패 메시지가 없습니다."
    else
        check_fail "로그에서 에러 메시지가 발견됐습니다. 아래 내용을 확인하세요:"
        echo "$ERROR_LINES" | sed 's/^/    /'
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
# 3. 체크포인트 파일이 정상적으로 정리됐는지 (오염 방지 확인)
# -------------------------------------------------------------
echo "[3] 체크포인트 정리 상태 확인"

CHECKPOINT_DIR="/app/data/checkpoints"
LEFTOVER_CHECKPOINTS=$(ls "$CHECKPOINT_DIR"/extraction_*_checkpoint.json 2>/dev/null)

if [ -z "$LEFTOVER_CHECKPOINTS" ]; then
    check_pass "임시 체크포인트 파일이 정상적으로 삭제되었습니다."
else
    check_fail "삭제되지 않은 체크포인트 파일이 남아있습니다. 다음 실행 시 오염 위험이 있으니 확인하세요:"
    echo "$LEFTOVER_CHECKPOINTS" | sed 's/^/    /'
fi
echo ""

# -------------------------------------------------------------
# 4. 실제로 데이터가 늘어났는지 (ChromaDB 문서 수 확인)
# -------------------------------------------------------------
echo "[4] ChromaDB 데이터 적재 현황"

python3 -c "
from indigo_pipeline.stores.vector_store import ChromaVectorStore
store = ChromaVectorStore()
for name, collection in store.collections.items():
    print(f'    {name}: {collection.count()}건')
" 2>/dev/null

if [ $? -eq 0 ]; then
    check_pass "ChromaDB 컬렉션별 건수를 정상적으로 조회했습니다. 위 수치를 이전 실행 시점과 비교해 늘어났는지 확인하세요."
else
    check_fail "ChromaDB 조회에 실패했습니다. 컨테이너 안에서 이 스크립트를 실행 중인지 확인하세요."
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
