#!/bin/bash
# =============================================================
# Indigo maintenance orchestrator
#
# Runs safe operational checks and repairs from the host repo.
# It never deletes rag_store data and never clears ChromaDB.
# =============================================================

set -u

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
LOG_DIR="${LOG_DIR:-$DATA_DIR/logs/maintenance}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/maintenance_${TIMESTAMP}.log"

API_SERVICE="${INDIGO_API_SERVICE:-api}"
NGINX_SERVICE="${INDIGO_NGINX_SERVICE:-nginx}"
API_CONTAINER="${INDIGO_API_CONTAINER:-search_api}"
NGINX_CONTAINER="${INDIGO_NGINX_CONTAINER:-nginx}"

DOC_TYPE="all"
RUN_INCREMENTAL_INDEXING=false
RUN_FULL_PIPELINE=false
RUN_RETRY_FAILED=false
CHECK_DUPLICATES=false
REPAIR=false
DRY_RUN=false
RETENTION_DAYS="${INDIGO_SPLIT_RUN_RETENTION_DAYS:-7}"
MAX_RUNS="${INDIGO_SPLIT_RUN_MAX_RUNS:-3}"
FAILURES=0

mkdir -p "$LOG_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash maintenance_orchestrator.sh [options]

Safe by default: checks only, no repair and no pipeline run.

Options:
  --repair                 Start missing api/nginx, remove temporary checkpoints, reload nginx.
  --incremental-indexing   Run split_runner incrementally. Existing Chroma doc_ids are skipped.
  --full-pipeline          Run the normal pipeline container entrypoint.
  --retry-failed           Retry failed GPT extraction docs using current train data.
  --check-duplicates       Check ChromaDB duplicate ids/doc_ids.
  --doc-type TYPE          all, patent, article, or project. Default: all.
  --retention-days N       split_index_runs retention days. Default: env or 7.
  --max-runs N             newest split_index_runs to keep. Default: env or 3.
  --dry-run                Print actions without executing mutating commands.
  -h, --help               Show this help.

Examples:
  bash maintenance_orchestrator.sh --repair
  bash maintenance_orchestrator.sh --repair --incremental-indexing --doc-type all
  bash maintenance_orchestrator.sh --repair --retry-failed --doc-type article
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --repair) REPAIR=true ;;
        --incremental-indexing) RUN_INCREMENTAL_INDEXING=true ;;
        --full-pipeline) RUN_FULL_PIPELINE=true ;;
        --retry-failed) RUN_RETRY_FAILED=true ;;
        --check-duplicates) CHECK_DUPLICATES=true ;;
        --doc-type)
            shift
            DOC_TYPE="${1:-all}"
            ;;
        --retention-days)
            shift
            RETENTION_DAYS="${1:-7}"
            ;;
        --max-runs)
            shift
            MAX_RUNS="${1:-3}"
            ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
    shift
done

case "$DOC_TYPE" in
    all|patent|article|project) ;;
    *)
        echo "Invalid --doc-type: $DOC_TYPE" | tee -a "$LOG_FILE"
        exit 2
        ;;
esac

log() {
    echo "[$(date)] $*" | tee -a "$LOG_FILE"
}

run_cmd() {
    log "+ $*"
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local code="${PIPESTATUS[0]}"
    if [ "$code" -ne 0 ]; then
        log "WARN: command failed with exit $code"
        FAILURES=$((FAILURES + 1))
    fi
    return "$code"
}

run_cmd_optional() {
    log "+ $*"
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    "$@" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

container_exists() {
    docker container inspect "$1" >/dev/null 2>&1
}

container_running() {
    docker ps --format '{{.Names}}' | grep -qx "$1"
}

ensure_api_stack() {
    if container_running "$API_CONTAINER" && container_running "$NGINX_CONTAINER"; then
        log "api/nginx containers are running."
        return 0
    fi

    log "api/nginx is not fully running. api=$(container_running "$API_CONTAINER" && echo up || echo down), nginx=$(container_running "$NGINX_CONTAINER" && echo up || echo down)"
    if [ "$REPAIR" = true ]; then
        run_cmd compose up -d "$API_SERVICE" "$NGINX_SERVICE"
    else
        log "repair disabled. Use --repair to start missing containers."
    fi
}

reload_nginx() {
    if ! container_exists "$NGINX_CONTAINER"; then
        log "nginx container not found; skip reload."
        return 0
    fi
    if [ "$REPAIR" != true ]; then
        log "repair disabled. Would reload nginx."
        return 0
    fi
    sleep 5
    run_cmd_optional docker exec "$NGINX_CONTAINER" nginx -s reload || run_cmd docker restart "$NGINX_CONTAINER"
}

cleanup_temp_checkpoints() {
    local checkpoint_dir="$DATA_DIR/checkpoints"
    if [ ! -d "$checkpoint_dir" ]; then
        log "checkpoint dir not found: $checkpoint_dir"
        return 0
    fi

    local files
    files=$(find "$checkpoint_dir" -maxdepth 1 -name 'extraction_*_checkpoint.json' -type f 2>/dev/null || true)
    if [ -z "$files" ]; then
        log "no temporary extraction checkpoints found."
        return 0
    fi

    log "temporary extraction checkpoints found:"
    echo "$files" | tee -a "$LOG_FILE"
    if [ "$REPAIR" = true ]; then
        if [ "$DRY_RUN" = true ]; then
            log "+ rm -f $files"
        else
            echo "$files" | xargs -r rm -f
            log "temporary extraction checkpoints removed."
        fi
    else
        log "repair disabled. Use --repair to remove temporary checkpoints."
    fi
}

show_store_counts() {
    log "checking ChromaDB collection counts."
    run_cmd compose --profile pipeline run --rm pipeline python3 - <<'PY'
from indigo_pipeline.stores import ChromaVectorStore

store = ChromaVectorStore()
for name in [
    "patent_chunks", "patent_entities", "patent_relations",
    "article_chunks", "article_entities", "article_relations",
    "project_chunks", "project_entities", "project_relations",
]:
    print(name, store.collections[name].count())
PY
}

check_duplicates() {
    log "checking ChromaDB duplicate ids/doc_ids."
    run_cmd compose --profile pipeline run --rm pipeline python3 - <<'PY'
from collections import Counter
from indigo_pipeline.stores import ChromaVectorStore

DOC_TYPES = ["patent", "article", "project"]
ITEM_TYPES = ["chunks", "entities", "relations"]
BATCH = 2000

store = ChromaVectorStore()
found = False

for doc_type in DOC_TYPES:
    print(f"[{doc_type}]")
    for item_type in ITEM_TYPES:
        name = f"{doc_type}_{item_type}"
        collection = store.collections.get(name)
        if collection is None:
            continue
        total = collection.count()
        id_counter = Counter()
        doc_id_counter = Counter()
        offset = 0
        while offset < total:
            result = collection.get(include=["metadatas"], limit=BATCH, offset=offset)
            id_counter.update(result.get("ids", []))
            if item_type == "chunks":
                doc_id_counter.update(m.get("doc_id") for m in result.get("metadatas", []) if m)
            offset += BATCH

        dup_ids = {key: count for key, count in id_counter.items() if count > 1}
        if dup_ids:
            print(f"  WARN {name}: duplicate collection ids={len(dup_ids)}")
            found = True
        else:
            print(f"  OK   {name}: no duplicate collection ids, total={sum(id_counter.values())}")

        if item_type == "chunks":
            dup_doc_ids = {key: count for key, count in doc_id_counter.items() if count > 1}
            if dup_doc_ids:
                print(f"  WARN {name}: duplicate doc_ids={len(dup_doc_ids)}")
                found = True
            else:
                print(f"  OK   {name}: no duplicate doc_ids, unique_docs={len(doc_id_counter)}")

raise SystemExit(2 if found else 0)
PY
}

run_incremental_indexing() {
    log "running incremental indexing for doc_type=$DOC_TYPE"
    run_cmd compose --profile pipeline run --rm pipeline \
        python3 -m indigo_pipeline.indexing.split_runner \
        --doc-type "$DOC_TYPE" \
        --llm-batch-docs 30 \
        --extract-batch-size 5 \
        --embedding-batch-size 50 \
        --resume-extract \
        --retries 1 \
        --keep-going \
        --lock-file /app/data/logs/split_runner.lock \
        --stale-lock-seconds 43200 \
        --cleanup-success \
        --retention-days "$RETENTION_DAYS" \
        --max-runs "$MAX_RUNS"
}

run_full_pipeline() {
    log "running full pipeline entrypoint."
    run_cmd compose --profile pipeline run --rm pipeline
}

retry_failed() {
    local types
    if [ "$DOC_TYPE" = "all" ]; then
        types="patent article project"
    else
        types="$DOC_TYPE"
    fi

    for type in $types; do
        log "retrying failed docs for doc_type=$type"
        run_cmd compose --profile pipeline run --rm pipeline \
            python3 -m indigo_pipeline.indexing.builder \
            --doc-type "$type" \
            --retry-failed
    done
}

log "Indigo maintenance started. log=$LOG_FILE"
log "mode: repair=$REPAIR incremental=$RUN_INCREMENTAL_INDEXING full_pipeline=$RUN_FULL_PIPELINE retry_failed=$RUN_RETRY_FAILED check_duplicates=$CHECK_DUPLICATES doc_type=$DOC_TYPE dry_run=$DRY_RUN"

ensure_api_stack
cleanup_temp_checkpoints
show_store_counts

if [ "$CHECK_DUPLICATES" = true ]; then
    check_duplicates
fi

if [ "$RUN_RETRY_FAILED" = true ]; then
    retry_failed
fi

if [ "$RUN_INCREMENTAL_INDEXING" = true ]; then
    run_incremental_indexing
fi

if [ "$RUN_FULL_PIPELINE" = true ]; then
    run_full_pipeline
fi

if [ "$REPAIR" = true ]; then
    ensure_api_stack
    reload_nginx
fi

if [ "$FAILURES" -gt 0 ]; then
    log "Indigo maintenance finished with $FAILURES failure(s)."
    exit 1
fi

log "Indigo maintenance finished successfully."
