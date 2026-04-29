from django.apps import AppConfig
import sys

class SearchConfig(AppConfig):
    name = 'search'

    def ready(self):
        management_commands_without_search_runtime = {
            "build_rag_index",
            "collect_rag_sources",
            "clear_search_result_cache",
            "collectstatic",
            "filter_rag_data",
            "makemigrations",
            "migrate",
            "shell",
            "test",
        }
        if len(sys.argv) > 1 and sys.argv[1] in management_commands_without_search_runtime:
            return

        from .engine.embedder import Embedder
        Embedder()
