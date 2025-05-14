"""Gene RAG: A retrieval-augmented generation system for genetic research."""

from .core import GeneRAG
from .config import Config

__version__ = "0.1.1"
__all__ = ["GeneRAG", "Config"]