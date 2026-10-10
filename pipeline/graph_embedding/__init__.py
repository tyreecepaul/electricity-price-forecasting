"""Learned cross-region embeddings appended to the prepared model features.

The encoder itself (``embedder``) imports torch and is loaded lazily by the
experiment pipeline, so importing this package stays cheap.
"""

from .config import DEFAULT_NODE_FEATURES, GraphEmbeddingConfig
from .graph import DEFAULT_REGIONS, NEM_INTERCONNECTORS, adjacency

__all__ = [
    "DEFAULT_NODE_FEATURES",
    "DEFAULT_REGIONS",
    "GraphEmbeddingConfig",
    "NEM_INTERCONNECTORS",
    "adjacency",
]
