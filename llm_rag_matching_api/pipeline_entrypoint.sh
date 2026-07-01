#!/bin/bash
# =============================================================
# Indigo Pipeline Entrypoint
# 수집 → 필터링 → GPT 추출 → 임베딩 저장 전체 파이프라인
# 실행 중 API 컨테이너 중지 → 완료 후 재시작
# =============================================================

set -eu

LOG_DIR="/app/data/logs/pipeline"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "[$(date)] Pipeline started" | tee -a "$LOG_FILE"

API_CONTAINER="${INDIGO_API_CONTAINER:-search_api}"

container_exists() {
    docker container inspect "$1" >/dev/null 2>&1
}

stop_api_container() {
    if container_exists "$API_CONTAINER"; then
        docker stop "$API_CONTAINER" || true
        echo "[$(date)] API container stopped: $API_CONTAINER" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] API container not found, skip stop: $API_CONTAINER" | tee -a "$LOG_FILE"
    fi
}

start_api_container() {
    if container_exists "$API_CONTAINER"; then
        docker start "$API_CONTAINER" || true
        echo "[$(date)] API container restart requested: $API_CONTAINER" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] API container not found, skip restart: $API_CONTAINER" | tee -a "$LOG_FILE"
    fi
}

# =============================================================
# API 컨테이너 중지 (GPU 전체 확보)
# =============================================================
echo "[$(date)] Stopping API container..." | tee -a "$LOG_FILE"
stop_api_container

# =============================================================
# 1. 수집 + 필터링 (collection_runner)
# =============================================================
echo "[$(date)] Step 1: Collection + Filtering" | tee -a "$LOG_FILE"

python -m indigo_pipeline.collection_runner \
    --profile all \
    --resume \
    --retries 2 \
    --retry-delay 120 \
    --keep-going \
    --step-timeout 21600 \
    --lock-file /app/data/logs/collection_runner.lock \
    --stale-lock-seconds 86400 \
    --log-dir "$LOG_DIR/collection" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] Collection failed (exit $EXIT_CODE). Restarting API and aborting." | tee -a "$LOG_FILE"
    start_api_container
    exit $EXIT_CODE
fi

echo "[$(date)] Collection complete." | tee -a "$LOG_FILE"

# =============================================================
# 2. GPT 추출 + 임베딩 저장 (split_runner)
# =============================================================
echo "[$(date)] Step 2: Indexing (split_runner)" | tee -a "$LOG_FILE"

python -m indigo_pipeline.indexing.split_runner \
    --doc-type all \
    --llm-batch-docs 30 \
    --extract-batch-size 5 \
    --embedding-batch-size 50 \
    --resume-extract \
    --retries 1 \
    --keep-going \
    --lock-file /app/data/logs/split_runner.lock \
    --stale-lock-seconds 43200 \
    --cleanup-success \
    --retention-days 7 \
    --max-runs 3 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] Indexing failed (exit $EXIT_CODE). Restarting API and aborting." | tee -a "$LOG_FILE"
    start_api_container
    exit $EXIT_CODE
fi

echo "[$(date)] Indexing complete." | tee -a "$LOG_FILE"

# =============================================================
# API 컨테이너 재시작
# =============================================================
echo "[$(date)] Restarting API container..." | tee -a "$LOG_FILE"
start_api_container

echo "[$(date)] Pipeline finished." | tee -a "$LOG_FILE"
