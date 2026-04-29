import json

from django.core.management.base import BaseCommand, CommandError

from search.engine.index_builder import IncrementalIndexBuilder


class Command(BaseCommand):
    help = (
        "Incrementally build the RAG index from data/train JSON files. "
        "Documents already present in the Chroma chunk collection are skipped before preprocessing."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-type",
            required=True,
            choices=["patent", "article", "project"],
            help="Document type to index.",
        )
        parser.add_argument(
            "--data-file",
            default=None,
            help="Optional JSON file path. Defaults to the configured data/train file for doc-type.",
        )
        parser.add_argument(
            "--store-dir",
            default=None,
            help="Optional RAG store directory. Defaults to search.engine.settings.RAG_STORE_DIR.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of unique source documents to inspect.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,
            help="LLM extraction batch size.",
        )
        parser.add_argument(
            "--force-api",
            action="store_true",
            help="Force OpenAI embeddings instead of local GPU embeddings.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only report duplicate filtering stats. No preprocessing, LLM calls, embeddings, or writes.",
        )

    def handle(self, *args, **options):
        try:
            builder = IncrementalIndexBuilder(
                doc_type=options["doc_type"],
                force_api=options["force_api"],
                store_dir=options["store_dir"],
            )
            stats = builder.run(
                data_file=options["data_file"],
                limit=options["limit"],
                dry_run=options["dry_run"],
                batch_size=options["batch_size"],
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2))
        self.stdout.write(self.style.SUCCESS("RAG index command completed."))
