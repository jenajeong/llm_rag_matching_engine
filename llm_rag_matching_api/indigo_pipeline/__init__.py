"""Standalone Indigo train-data indexing pipeline."""

__all__ = ["IndexBuilder"]


def __getattr__(name):
    if name == "IndexBuilder":
        from .indexing.builder import IndexBuilder

        return IndexBuilder
    raise AttributeError(name)
