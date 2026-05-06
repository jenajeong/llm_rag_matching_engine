import argparse
import json

DOC_TYPES = ["patent", "article", "project"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Indigo train-data indexing pipeline.")
    parser.add_argument("--doc-type", choices=[*DOC_TYPES, "all"], default="all")
    parser.add_argument("--data-file", default=None, help="Only valid when --doc-type is not all.")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retry", type=int, default=None)
    parser.add_argument("--phase", choices=["full", "extract", "store"], default="full")
    parser.add_argument("--extraction-file", default=None, help="Only valid when --doc-type is not all.")
    parser.add_argument("--prepared-docs-file", default=None, help="Only valid with --phase extract and a single doc type.")
    return parser


def run_pipeline(args) -> list[dict]:
    results = []
    doc_types = DOC_TYPES if args.doc_type == "all" else [args.doc_type]
    if args.data_file and args.doc_type == "all":
        raise ValueError("--data-file can only be used with a single --doc-type.")
    if args.extraction_file and args.doc_type == "all":
        raise ValueError("--extraction-file can only be used with a single --doc-type.")
    if args.prepared_docs_file and args.doc_type == "all":
        raise ValueError("--prepared-docs-file can only be used with a single --doc-type.")
    for index, doc_type in enumerate(doc_types):
        from .indexing.builder import IndexBuilder, setup_logging

        setup_logging(doc_type)
        clear_for_doc = args.clear and index == 0
        builder = IndexBuilder(
            doc_type=doc_type,
            force_api=args.force_api,
            store_dir=args.store_dir,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        if args.retry_failed:
            result = builder.retry_failed(max_docs=args.max_retry)
        elif args.phase == "extract":
            result = builder.run_extract(
                data_file=args.data_file,
                clear=args.clear,
                batch_size=args.batch_size,
                output_file=args.extraction_file,
                prepared_docs_file=args.prepared_docs_file,
            )
        elif args.phase == "store":
            result = builder.run_store(extraction_file=args.extraction_file, clear=clear_for_doc)
        else:
            result = builder.run(
                data_file=args.data_file,
                clear=clear_for_doc,
                resume=args.resume,
                batch_size=args.batch_size,
            )
        results.append({"doc_type": doc_type, "result": result})
    return results


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
