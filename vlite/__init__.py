__all__ = ["VLite", "EmbeddingModel", "RustVLite", "open_local"]


def __getattr__(name):
    if name in {"VLite", "EmbeddingModel"}:
        if name == "VLite":
            from .main import VLite

            return VLite
        from .model import EmbeddingModel

        return EmbeddingModel

    if name in {"RustVLite", "open_local"}:
        try:
            from vlite_py import RustVLite, open_local
        except ImportError:  # pragma: no cover - optional during migration
            return None
        return {"RustVLite": RustVLite, "open_local": open_local}[name]

    raise AttributeError(f"module 'vlite' has no attribute {name!r}")