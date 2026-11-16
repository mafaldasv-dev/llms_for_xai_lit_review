"""Utility functions for normalization of paper metadata."""

import re
from typing import Optional


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize a DOI string by lowercasing and stripping common prefixes."""
    if not doi or str(doi).lower() == "nan":
        return None
    d = str(doi).lower().strip()
    for p in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
        if d.startswith(p):
            d = d[len(p):]
    return d.strip() or None


def normalize_title(title: Optional[str]) -> str:
    """Normalize a title by lowercasing and collapsing whitespace."""
    if not title or str(title).lower() == "nan":
        return ""
    t = str(title).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_author_name(name: str) -> str:
    """Normalize an author name by lowercasing and removing punctuation."""
    n = (name or "").lower()
    n = re.sub(r"[,.\-()]", " ", n)
    n = " ".join(n.split())
    return n


def _token_sort_key(s: str) -> str:
    """Sort tokens in a string alphabetically (for fuzzy matching)."""
    toks = [t for t in re.split(r"\s+", s) if t]
    return " ".join(sorted(toks))
