"""Načítavanie datasetov použitých vo finálnej verejnej verzii experimentov."""

from .flickr30k import load_flickr30k_karpathy
from .scicap import load_scicap

__all__ = ["load_flickr30k_karpathy", "load_scicap"]
