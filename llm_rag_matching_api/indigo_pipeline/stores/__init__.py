__all__ = ["ChromaVectorStore", "GraphStore"]


def __getattr__(name):
    if name == "ChromaVectorStore":
        from .vector_store import ChromaVectorStore

        return ChromaVectorStore
    if name == "GraphStore":
        from .graph_store import GraphStore

        return GraphStore
    raise AttributeError(name)
