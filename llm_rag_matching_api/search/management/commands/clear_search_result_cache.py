from django.core.management.base import BaseCommand

from search.engine.result_cache import clear_search_results
from search.engine.settings import SEARCH_RESULT_CACHE_TTL_HOURS


class Command(BaseCommand):
    help = "Clear cached recommendation results used by the report generation endpoint."

    def add_arguments(self, parser):
        parser.add_argument(
            "--older-than-hours",
            type=int,
            default=SEARCH_RESULT_CACHE_TTL_HOURS,
            help=f"Delete cache files older than this many hours. Default: {SEARCH_RESULT_CACHE_TTL_HOURS}.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Delete all cached recommendation results.",
        )

    def handle(self, *args, **options):
        deleted = clear_search_results(
            older_than_hours=options["older_than_hours"],
            clear_all=options["all"],
        )
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} cached search result file(s)."))
