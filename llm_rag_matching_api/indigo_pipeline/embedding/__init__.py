__all__ = ["Embedder"]


def __getattr__(name):
    if name == "Embedder":
        from .embedder import Embedder

        return Embedder
    raise AttributeError(name)
