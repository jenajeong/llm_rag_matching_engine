__all__ = ["EntityRelationExtractor", "AsyncEntityRelationExtractor"]


def __getattr__(name):
    if name in __all__:
        from .entity_extractor import AsyncEntityRelationExtractor, EntityRelationExtractor

        return {
            "EntityRelationExtractor": EntityRelationExtractor,
            "AsyncEntityRelationExtractor": AsyncEntityRelationExtractor,
        }[name]
    raise AttributeError(name)
