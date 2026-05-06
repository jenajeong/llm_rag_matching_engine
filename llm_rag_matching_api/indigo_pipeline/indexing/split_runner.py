import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import config
from .builder import IndexBuilder, setup_logging
from .merge import merge_duplicate_entities, merge_duplicate_relations


DOC_TYPES = ["patent", "article", "project"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GPT extraction in subprocess batches, then load embedding model once for store phase."
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
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-extract", action="store_true", help="Reuse the combined extraction artifact and only store.")
    parser.add_argument("--skip-store", action="store_true", help="Only run extraction subprocesses and artifact merge.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_suffix(f"{path.suffix}.tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _run_command(command: list[str], cwd: Path, retries: int) -> int:
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        print(f"Running attempt {attempt}/{attempts}: {' '.join(command)}")
        completed = subprocess.run(command, cwd=cwd)
        if completed.returncode == 0:
            return 0
    return completed.returncode


def _combine_artifacts(doc_type: str, artifact_files: list[Path], output_file: Path) -> dict[str, Any]:
    docs: list[dict] = []
    entities: list[dict] = []
    relations: list[dict] = []
    failed_doc_ids: list[str] = []
    stats: dict[str, Any] = {
        "docs_processed": 0,
        "docs_new": 0,
        "entities_extracted": 0,
        "relations_extracted": 0,
        "failed_docs": 0,
    }

    for artifact_file in artifact_files:
        artifact = _read_json(artifact_file)
        docs.extend(artifact.get("docs", []))
        entities.extend(artifact.get("entities", []))
        relations.extend(artifact.get("relations", []))
        failed_doc_ids.extend(artifact.get("failed_doc_ids", []))
        batch_stats = artifact.get("stats", {})
        for key in stats:
            stats[key] += int(batch_stats.get(key, 0) or 0)

    entities = merge_duplicate_entities(entities)
    relations = merge_duplicate_relations(relations)
    stats["entities_after_merge"] = len(entities)
    stats["relations_after_merge"] = len(relations)
    stats["failed_docs"] = len(set(failed_doc_ids))

    payload = {
        "doc_type": doc_type,
        "docs": docs,
        "entities": entities,
        "relations": relations,
        "failed_doc_ids": sorted(set(failed_doc_ids)),
        "stats": stats,
        "saved_at": datetime.now().isoformat(),
    }
    _write_json(output_file, payload)
    return payload


def _run_doc_type(args: argparse.Namespace, doc_type: str, run_root: Path, clear_store: bool) -> dict[str, Any]:
    setup_logging(f"split_{doc_type}")
    doc_run_dir = run_root / doc_type
    combined_file = doc_run_dir / f"{doc_type}_extraction_combined.json"

    if not args.skip_extract:
        builder = IndexBuilder(
            doc_type=doc_type,
            force_api=args.force_api,
            store_dir=args.store_dir,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        docs = builder.prepare_documents(data_file=args.data_file, clear=args.clear)
        doc_batches = _chunked(docs, max(1, args.llm_batch_docs))
        artifact_files: list[Path] = []

        print(f"{doc_type}: prepared {len(docs)} docs into {len(doc_batches)} extraction subprocess batches")
        for batch_index, batch_docs in enumerate(doc_batches, 1):
            docs_file = doc_run_dir / "docs" / f"batch_{batch_index:04d}.json"
            artifact_file = doc_run_dir / "artifacts" / f"batch_{batch_index:04d}.json"
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

        combined = _combine_artifacts(doc_type, artifact_files, combined_file)
    else:
        combined = _read_json(combined_file)

    if not args.skip_store:
        store_builder = IndexBuilder(
            doc_type=doc_type,
            force_api=args.force_api,
            store_dir=args.store_dir,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        store_result = store_builder.run_store(extraction_file=combined_file, clear=clear_store)
    else:
        store_result = None

    return {
        "doc_type": doc_type,
        "combined_artifact": str(combined_file),
        "docs": len(combined.get("docs", [])),
        "entities": len(combined.get("entities", [])),
        "relations": len(combined.get("relations", [])),
        "store_result": store_result,
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.data_file and args.doc_type == "all":
        raise ValueError("--data-file can only be used with a single --doc-type.")
    run_root = Path(args.run_dir) if args.run_dir else config.CHECKPOINT_DIR / "split_index_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_types = DOC_TYPES if args.doc_type == "all" else [args.doc_type]
    return [_run_doc_type(args, doc_type, run_root, clear_store=args.clear and index == 0) for index, doc_type in enumerate(doc_types)]


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
