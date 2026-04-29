import json

from django.core.management.base import BaseCommand, CommandError

from search.engine.filtering import filter_raw_data
from search.engine.json_store import load_json_list
from search.engine.filtering import TRAIN_FILES


class Command(BaseCommand):
    help = "Filter collected raw RAG data into data/train JSON files using the existing data flow."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-type",
            required=True,
            choices=["patent", "article", "project"],
            help="Document type to filter.",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            help="Write data/train output. Without this flag the command is dry-run only.",
        )
        parser.add_argument(
            "--allow-count-decrease",
            action="store_true",
            help="Allow overwriting an existing train file with fewer records.",
        )

    def handle(self, *args, **options):
        try:
            stats = filter_raw_data(
                doc_type=options["doc_type"],
                dry_run=True,
            )
            if options["write"]:
                train_file = TRAIN_FILES[options["doc_type"]]
                existing_count = len(load_json_list(train_file))
                if (
                    existing_count
                    and stats["filtered_count"] < existing_count
                    and not options["allow_count_decrease"]
                ):
                    raise CommandError(
                        "Refusing to overwrite train data with fewer records "
                        f"({stats['filtered_count']} < {existing_count}). "
                        "Use --allow-count-decrease only after manually verifying the change."
                    )
                stats = filter_raw_data(
                    doc_type=options["doc_type"],
                    dry_run=False,
                )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2))
        self.stdout.write(self.style.SUCCESS("RAG filtering command completed."))
