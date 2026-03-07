from .main import VLite
from .model import EmbeddingModel

try:
    from vlite_py import RustVLite, open_local
except ImportError:  # pragma: no cover - optional during migration
    RustVLite = None
    open_local = None

__all__ = ["VLite", "EmbeddingModel", "RustVLite", "open_local"]