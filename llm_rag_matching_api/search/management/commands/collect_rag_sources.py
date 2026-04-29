import json

from django.core.management.base import BaseCommand, CommandError

from search.engine.collectors import ArticleCollector, PatentCollector, ProjectCollector


class Command(BaseCommand):
    help = "Safely collect Indigo source data into existing raw JSON files without appending duplicates."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-type",
            required=True,
            choices=["patent", "article", "project"],
            help="Source document type to collect.",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            help="Append new raw records. Without this flag the command is dry-run only.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit DB candidates for testing.",
        )
        parser.add_argument(
            "--sleep-seconds",
            type=float,
            default=1.0,
            help="Delay between KIPRIS API calls for patent collection.",
        )
        parser.add_argument(
            "--project-source-1",
            default=None,
            help="Optional project_source_1.json path.",
        )
        parser.add_argument(
            "--project-source-2",
            default=None,
            help="Optional project_source_2.json path.",
        )

    def handle(self, *args, **options):
        doc_type = options["doc_type"]
        dry_run = not options["write"]

        try:
            if doc_type == "patent":
                stats = PatentCollector().collect(
                    limit=options["limit"],
                    dry_run=dry_run,
                    sleep_seconds=options["sleep_seconds"],
                )
            elif doc_type == "article":
                stats = ArticleCollector().collect_candidates(
                    limit=options["limit"],
                    dry_run=dry_run,
                )
                if options["write"]:
                    stats["note"] = (
                        "Article EBSCO crawling is not executed by this command. "
                        "Use this output to inspect safe DB candidates before running a crawler."
                    )
            else:
                stats = ProjectCollector(
                    json_file1=options["project_source_1"],
                    json_file2=options["project_source_2"],
                ).collect(
                    limit=options["limit"],
                    dry_run=dry_run,
                )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2))
        self.stdout.write(self.style.SUCCESS("RAG source collection command completed."))
