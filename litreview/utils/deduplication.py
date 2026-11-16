"""Utility functions and classes for deduplication of papers."""

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List

from litreview.utils.normalization import _token_sort_key, normalize_title

try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def find_doi_duplicates(
    records: List[Dict[str, Any]],
    doi_key: str = "normalised_doi",
) -> List[List[Dict[str, Any]]]:
    """Group records that share the same normalized DOI.

    Returns a list of groups, where each group has 2+ records with the same DOI.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        d = r.get(doi_key)
        if d and str(d).lower() != "nan":
            groups[d].append(r)
    return [g for g in groups.values() if len(g) > 1]


class TitleMatcher:
    """Greedy grouping by fuzzy title similarity.

    Parameters
    ----------
    threshold : int
        Similarity threshold in 0-100 (like rapidfuzz).
        Fallback uses SequenceMatcher ratio * 100.
    """

    def __init__(self, threshold: int = 90):
        self.threshold = threshold

    def _similarity(self, t1: str, t2: str) -> float:
        if not t1 or not t2:
            return 0.0
        k1 = _token_sort_key(normalize_title(t1))
        k2 = _token_sort_key(normalize_title(t2))
        if _HAS_RAPIDFUZZ:
            return fuzz.token_sort_ratio(k1, k2)
        return SequenceMatcher(None, k1, k2).ratio() * 100.0

    def find_duplicates(
        self, records: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Find groups of records with similar titles."""
        remaining = records.copy()
        groups = []
        while remaining:
            base = remaining.pop(0)
            base_title = base.get("Title", "") or ""
            current = [base]
            to_remove_idx = []
            for i, r in enumerate(remaining):
                score = self._similarity(base_title, r.get("Title", "") or "")
                if score >= self.threshold:
                    current.append(r)
                    to_remove_idx.append(i)
            for i in reversed(to_remove_idx):
                remaining.pop(i)
            if len(current) > 1:
                groups.append(current)
        return groups

    def get_confidence(self, group: List[Dict[str, Any]]) -> float:
        """Return average pairwise similarity for a group (0.0 to 1.0)."""
        if len(group) < 2:
            return 0.0
        sims = []
        titles = [
            normalize_title(r.get("normalised_title", "") or "") for r in group
        ]
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                sims.append(self._similarity(titles[i], titles[j]))
        avg = sum(sims) / len(sims) if sims else 0.0
        return avg / 100.0
