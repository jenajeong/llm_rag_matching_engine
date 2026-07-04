import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .. import config
from .builder import IndexBuilder, setup_logging


DOC_TYPES = ["patent", "article", "project"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GPT extraction in subprocess batches, then stream artifacts into one embedding/store phase."
    )
    parser.add_argument("--doc-type", choices=[*DOC_TYPES, "all"], default="all")
    parser.add_argument("--data-file", default=None, help="Only valid with a single --doc-type.")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--llm-batch-docs", type=int, default=50, help="Number of documents per extraction subprocess.")
    parser.add_argument("--extract-batch-size", type=int, default=10, help="Batch size inside each extraction subprocess.")
    parser.add_argument("--embedding-batch-size", type=int, default=128, help="Embedding/upsert batch size in store phase.")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--resume-extract",
        action="store_true",
        help="Skip extraction batches with valid artifacts. Use only for the same run-dir with unchanged data/code.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Reuse manifest and only store.")
    parser.add_argument("--skip-store", action="store_true", help="Only run extraction subprocesses and manifest creation.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lock-file", default=str(config.LOG_DIR / "split_runner.lock"))
    parser.add_argument("--wait-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=int, default=0, help="Seconds to wait for lock when --wait-lock is set. 0 means forever.")
    parser.add_argument("--stale-lock-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--cleanup-success", action="store_true", help="Delete per-batch docs/artifacts after successful store.")
    parser.add_argument(
        "--keep-extraction-checkpoints",
        action="store_true",
        help="Keep legacy extraction_{doc_type}_checkpoint.json files after a successful doc-type run.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=config.SPLIT_RUN_RETENTION_DAYS,
        help="Delete old split run directories after success. Default: INDIGO_SPLIT_RUN_RETENTION_DAYS.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=config.SPLIT_RUN_MAX_RUNS,
        help="Keep only the newest N split run directories after success. Default: INDIGO_SPLIT_RUN_MAX_RUNS.",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_suffix(f"{path.suffix}.tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size or 1))
    return [items[index:index + size] for index in range(0, len(items), size)]


def _run_command(command: list[str], cwd: Path, retries: int) -> int:
    attempts = retries + 1
    last_returncode = 1
    for attempt in range(1, attempts + 1):
        print(f"Running attempt {attempt}/{attempts}: {' '.join(command)}")
        completed = subprocess.run(command, cwd=cwd)
        last_returncode = completed.returncode
        if completed.returncode == 0:
            return 0
    return last_returncode


def _doc_signature(doc: dict[str, Any]) -> tuple[str, str]:
    doc_id = str(doc.get("doc_id") or "")
    text = " ".join(str(doc.get("text") or "").split())
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else ""
    return doc_id, text_hash


def _artifact_is_valid(path: Path, doc_type: str, expected_docs: list[dict[str, Any]] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        artifact = _read_json(path)
    except Exception:
        return False
    if artifact.get("doc_type") != doc_type:
        return False
    docs = artifact.get("docs")
    if not isinstance(docs, list) or not isinstance(artifact.get("entities"), list) or not isinstance(artifact.get("relations"), list):
        return False
    if expected_docs is None:
        return True
    return [_doc_signature(doc) for doc in docs] == [_doc_signature(doc) for doc in expected_docs]


def _manifest_is_valid(path: Path, doc_type: str) -> bool:
    if not path.exists():
        return False
    try:
        manifest = _read_json(path)
    except Exception:
        return False
    if manifest.get("doc_type") != doc_type:
        return False
    return all(_artifact_is_valid(Path(item), doc_type) for item in manifest.get("artifact_files", []))


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_stale(lock_file: Path, stale_lock_seconds: int) -> bool:
    try:
        payload = _read_json(lock_file)
        hostname = str(payload.get("hostname") or "")
        current_hostname = socket.gethostname()
        if hostname and hostname != current_hostname:
            print(f"Lock belongs to another container. Treating as stale: hostname={hostname}, current={current_hostname}, lock={lock_file}")
            return True

        pid = int(payload.get("pid") or 0)
        if pid and not _pid_is_running(pid):
            print(f"Lock PID is not running. Treating as stale: pid={pid}, lock={lock_file}")
            return True
    except Exception:
        pass

    try:
        age = time.time() - lock_file.stat().st_mtime
        return age > stale_lock_seconds
    except OSError:
        return False


def _create_manifest(doc_type: str, artifact_files: list[Path], output_file: Path) -> dict[str, Any]:
    manifest = {
        "doc_type": doc_type,
        "artifact_count": len(artifact_files),
        "artifact_files": [str(path) for path in artifact_files],
        "saved_at": datetime.now().isoformat(),
    }
    _write_json(output_file, manifest)
    return manifest


@contextmanager
def _runner_lock(args: argparse.Namespace):
    lock_file = Path(args.lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "started_at": datetime.now().isoformat(),
                        "command": sys.argv,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            break
        except FileExistsError:
            stale = _lock_is_stale(lock_file, args.stale_lock_seconds)
            if stale:
                print(f"Removing stale lock: {lock_file}")
                lock_file.unlink(missing_ok=True)
                continue
            if not args.wait_lock:
                raise RuntimeError(f"Another split runner appears to be running: {lock_file}")
            if args.lock_timeout and time.time() - started > args.lock_timeout:
                raise TimeoutError(f"Timed out waiting for lock: {lock_file}")
            print(f"Waiting for lock: {lock_file}")
            time.sleep(10)
    try:
        yield
    finally:
        lock_file.unlink(missing_ok=True)


def _cleanup_run_dir(doc_run_dir: Path) -> None:
    shutil.rmtree(doc_run_dir / "docs", ignore_errors=True)
    shutil.rmtree(doc_run_dir / "artifacts", ignore_errors=True)


def _cleanup_old_runs(run_root: Path, retention_days: int | None, max_runs: int | None) -> None:
    base_dir = run_root.parent
    if not base_dir.exists():
        return
    run_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if retention_days is not None:
        cutoff = datetime.now() - timedelta(days=retention_days)
        for path in run_dirs:
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
    if max_runs is not None and max_runs >= 0:
        refreshed = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for path in refreshed[max_runs:]:
            shutil.rmtree(path, ignore_errors=True)


def _cleanup_extraction_checkpoint(doc_type: str) -> None:
    checkpoint_file = config.CHECKPOINT_DIR / f"extraction_{doc_type}_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink(missing_ok=True)
        print(f"{doc_type}: removed legacy extraction checkpoint: {checkpoint_file}")


def _run_doc_type(args: argparse.Namespace, doc_type: str, run_root: Path, clear_store: bool) -> dict[str, Any]:
    setup_logging(f"split_{doc_type}")
    doc_run_dir = run_root / doc_type
    manifest_file = doc_run_dir / "manifest.json"

    if not args.skip_extract:
        builder = IndexBuilder(
            doc_type=doc_type,
            force_api=args.force_api,
            store_dir=args.store_dir,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        docs = builder.prepare_documents(data_file=args.data_file, clear=args.clear)
        doc_batches = _chunked(docs, args.llm_batch_docs)
        artifact_files: list[Path] = []

        print(f"{doc_type}: prepared {len(docs)} docs into {len(doc_batches)} extraction subprocess batches")
        for batch_index, batch_docs in enumerate(doc_batches, 1):
            docs_file = doc_run_dir / "docs" / f"batch_{batch_index:04d}.json"
            artifact_file = doc_run_dir / "artifacts" / f"batch_{batch_index:04d}.json"

            if args.resume_extract:
                if _artifact_is_valid(artifact_file, doc_type, expected_docs=batch_docs):
                    print(f"{doc_type}: skipping completed extraction batch {batch_index}")
                    artifact_files.append(artifact_file)
                    continue
                if artifact_file.exists():
                    print(f"{doc_type}: existing artifact does not match current batch {batch_index}; re-extracting")

            _write_json(docs_file, batch_docs)
            command = [
                args.python,
                "-m",
                "indigo_pipeline.indexing.builder",
                "--doc-type",
                doc_type,
                "--phase",
                "extract",
                "--prepared-docs-file",
                str(docs_file),
                "--extraction-file",
                str(artifact_file),
                "--batch-size",
                str(args.extract_batch_size),
                "--concurrency",
                str(args.concurrency),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
            ]
            returncode = _run_command(command, config.DJANGO_PROJECT_DIR, args.retries)
            if returncode != 0:
                if not args.keep_going:
                    raise RuntimeError(f"{doc_type} extraction batch {batch_index} failed with {returncode}")
                print(f"{doc_type} extraction batch {batch_index} failed with {returncode}; continuing")
                continue
            artifact_files.append(artifact_file)

        manifest = _create_manifest(doc_type, artifact_files, manifest_file)
    else:
        if not _manifest_is_valid(manifest_file, doc_type):
            raise FileNotFoundError(f"Valid manifest not found for --skip-extract: {manifest_file}")
        manifest = _read_json(manifest_file)

    store_result = None
    if not args.skip_store:
        store_builder = IndexBuilder(
            doc_type=doc_type,
            force_api=args.force_api,
            store_dir=args.store_dir,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        store_result = store_builder.run_store_manifest(
            manifest_file=manifest_file,
            clear=clear_store,
            embedding_batch_size=args.embedding_batch_size,
        )
        if args.cleanup_success:
            _cleanup_run_dir(doc_run_dir)

    if not args.keep_extraction_checkpoints:
        _cleanup_extraction_checkpoint(doc_type)

    return {
        "doc_type": doc_type,
        "manifest": str(manifest_file),
        "artifact_count": manifest.get("artifact_count", 0),
        "store_result": store_result,
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data_file and args.doc_type == "all":
        raise ValueError("--data-file can only be used with a single --doc-type.")
    run_root = Path(args.run_dir) if args.run_dir else config.CHECKPOINT_DIR / "split_index_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_types = DOC_TYPES if args.doc_type == "all" else [args.doc_type]
    with _runner_lock(args):
        results = [
            _run_doc_type(args, doc_type, run_root, clear_store=args.clear and index == 0)
            for index, doc_type in enumerate(doc_types)
        ]
        if args.retention_days is not None or args.max_runs is not None:
            _cleanup_old_runs(run_root, args.retention_days, args.max_runs)
        return results


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
