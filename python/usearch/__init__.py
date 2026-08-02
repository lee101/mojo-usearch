"""Mojo-backed exact dense-vector search compatible with the covered usearch API."""

from .index import BatchMatches, Index, Matches, MetricKind, ScalarKind, search

__version__ = "0.1.0"

__all__ = ["BatchMatches", "Index", "Matches", "MetricKind", "ScalarKind", "search"]
