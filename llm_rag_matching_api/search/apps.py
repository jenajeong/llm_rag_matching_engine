from django.apps import AppConfig

class SearchConfig(AppConfig):
    name = 'search'

    def ready(self):
        from .engine.embedder import Embedder
        Embedder()