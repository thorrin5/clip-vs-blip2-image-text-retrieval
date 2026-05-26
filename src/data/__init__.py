"""Dataset loaders for canonical retrieval experiments."""

from .ai2d import load_ai2d
from .flickr30k import load_flickr30k_karpathy
from .scicap import load_scicap

__all__ = ["load_ai2d", "load_flickr30k_karpathy", "load_scicap"]

