__all__ = ["IndexBuilder", "setup_logging"]


def __getattr__(name):
    if name in {"IndexBuilder", "setup_logging"}:
        from .builder import IndexBuilder, setup_logging

        return {"IndexBuilder": IndexBuilder, "setup_logging": setup_logging}[name]
    raise AttributeError(name)
