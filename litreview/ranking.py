"""Functions for loading ranking data and tagging papers by venue quality."""

import re
import time
from typing import List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz as _fuzz
    print("Using rapidfuzz for fuzzy string matching.")
    _HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher as _SM
    print("rapidfuzz not installed — falling back to difflib for fuzzy matching.")
    _HAS_RAPIDFUZZ = False


CONFERENCE_TYPES = [
    "inproceedings",
    "IEEE Conferences",
    "conference",
    "IET Conferences",
]
JOURNAL_TYPES = [
    "article",
    "IEEE Journals",
    "inbook",
    "TUP Journals",
    "ZUP Journals",
    "inproceedings",
]

# Ranking flag columns added to the DataFrame
RANKING_COLUMNS = [
    "is_core_conference_A*_A_B",
    "is_core_journal_A*_A_B",
    "is_scimago_q1_q2_2022",
    "scimago_2022_area",
    "is_scimago_q1_q2_2023",
    "scimago_2023_area",
    "is_scimago_q1_q2_2024",
    "scimago_2024_area",
    "is_predatory_journal_2025",
    "is_predatory_publisher_2025",
]

def _normalize_issn(issn: str) -> str:
    """Strip dashes and whitespace from an ISSN so it can be compared to
    SciMago's dash-free format (e.g. '1542-4863' → '15424863')."""
    return re.sub(r"[\s\-]", "", issn).lower()


def _fuzzy_score(a: str, b: str) -> float:
    """Return a 0-100 similarity score between two strings."""
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _fuzz.token_sort_ratio(a.lower(), b.lower())
    return _SM(None, a.lower(), b.lower()).ratio() * 100.0


def enrich_missing_venues(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Fix 3 — For rows with no usable venue name, query OpenAlex by DOI
    to fill in Source title and ISSN.

    Parameters
    ----------
    df : pd.DataFrame
        Papers DataFrame with at least 'Source title', 'Journal',
        'Book title', 'normalised_doi', 'ISSN' columns.
    api_key : str, optional
        OpenAlex API key for higher rate limits
        (set via pyalex.config.api_key).
    email : str, optional
        Polite-pool email for OpenAlex when no API key is used
        (set via pyalex.config.email).
    sleep_s : float
        Seconds to wait between API calls (default 0.2).

    Returns
    -------
    pd.DataFrame
        Copy of df with enriched venue data where previously missing.
    """
    try:
        import pyalex
        from pyalex import Works
        print("pyalex imported successfully for OpenAlex enrichment.")
    except ImportError:
        print("pyalex not installed — skipping OpenAlex enrichment.")
        return df

    if api_key:
        pyalex.config.api_key = api_key
    elif email:
        pyalex.config.email = email

    result = df.copy()

    # Only query rows that truly have no venue across all three fields:
    no_venue_mask = result[["Source title", "Journal", "Book title"]].apply(
        lambda col: col.astype(str).str.strip().isin(["", "nan", "NaN"])
    ).all(axis=1)
    candidates = result[no_venue_mask & result["normalised_doi"].notna()]

    if len(candidates) == 0:
        print("No records with missing venue and valid DOI found for OpenAlex enrichment.")
        return result

    print(f"Starting OpenAlex enrichment for {len(candidates)} records with missing venue...")

    for idx, row in candidates.iterrows():
        doi = str(row["normalised_doi"]).strip()
        try:
            work = Works()[doi]
            location = (work.get("primary_location") or {})
            source = (location.get("source") or {})
            venue_name = source.get("display_name") or ""
            issns = source.get("issn") or []
            issn = issns[0] if issns else ""

            if venue_name:
                result.loc[idx, "Source title"] = venue_name
            if issn and str(row.get("ISSN", "")).strip() in ("", "nan", "NaN"):
                result.loc[idx, "ISSN"] = issn

            time.sleep(sleep_s)

        except Exception as e:
            print(f"Error for DOI {doi}: {e} — skipping.")
            continue

    enriched = (result["Source title"] != df["Source title"]).sum()
    print(f"OpenAlex enrichment: filled venue for {enriched} records.")
    return result


def load_ranking_data(ranks_dir: str) -> dict:
    """Load all ranking CSV files from a directory.

    Returns a dict with keys:
        core_conf, core_jour, scimago_2022, scimago_2023, scimago_2024,
        pred_journals, pred_publishers
    """
    core_conf = pd.read_csv(
        f"{ranks_dir}/CORE_conferences_2023.csv", header=None
    )
    core_conf.columns = [
        "ID", "Conference Name", "Abbreviation",
        "CORE Year", "CORE Rank", "col6", "col7", "col8", "col9",
    ]

    core_jour = pd.read_csv(f"{ranks_dir}/CORE_journals_2020.csv")

    scimago_2022 = pd.read_csv(
        f"{ranks_dir}/scimagojr_2022_journal_ranks.csv",
        on_bad_lines="warn",
        delimiter=";",
    )
    scimago_2023 = pd.read_csv(
        f"{ranks_dir}/scimagojr_2023_journal_ranks.csv",
        on_bad_lines="warn",
        delimiter=";",
    )
    scimago_2024 = pd.read_csv(
        f"{ranks_dir}/scimagojr_2024_journal_ranks.csv",
        on_bad_lines="warn",
        delimiter=";",
    )

    pred_journals = pd.read_csv(
        f"{ranks_dir}/The Predatory Journals List 2025 - Sheet1.csv",
        header=None,
    )
    pred_journals.columns = ["some_index", "journal_name"]

    pred_publishers = pd.read_csv(
        f"{ranks_dir}/The Predatory Publishers List 2025 - Sheet1.csv",
        header=None,
    )
    pred_publishers.columns = ["some_index", "publisher_name"]

    print(f"Loaded ranking data from {ranks_dir}:")
    print(f"  CORE conferences 2023: {len(core_conf)} entries")
    print(f"  CORE journals 2020:    {len(core_jour)} entries")
    print(f"  SciMago 2022:          {len(scimago_2022)} entries")
    print(f"  SciMago 2023:          {len(scimago_2023)} entries")
    print(f"  SciMago 2024:          {len(scimago_2024)} entries")
    print(f"  Predatory journals:    {len(pred_journals)} entries")
    print(f"  Predatory publishers:  {len(pred_publishers)} entries")

    return {
        "core_conf": core_conf,
        "core_jour": core_jour,
        "scimago_2022": scimago_2022,
        "scimago_2023": scimago_2023,
        "scimago_2024": scimago_2024,
        "pred_journals": pred_journals,
        "pred_publishers": pred_publishers,
    }


def check_in_core_conferences(
    names_to_match: List[str],
    core_conf_df: pd.DataFrame,
    fuzzy_threshold: int = 85,
) -> str:
    """Check if any of the given names match a CORE conference. Returns rank or ''.

    Matching order (Fix 2):
    1. Exact match on Conference Name (original behaviour).
    2. Exact match on Abbreviation — catches cases where Source title contains
       only the acronym (e.g. 'AAAI', 'COLING', 'CIKM').
    3. Fuzzy match on Conference Name — catches minor wording differences
       (e.g. 'Proceedings of the AAAI Conference...' vs the CORE entry name).
       The best-scoring candidate above ``fuzzy_threshold`` is returned.
    """
    abbrev_lower = core_conf_df["Abbreviation"].str.strip().str.lower()
    name_lower = core_conf_df["Conference Name"].str.strip().str.lower()

    for conf_name in names_to_match:
        q = conf_name.strip().lower()
        if not q or q == "nan":
            continue

        # 1. Exact name match
        matched = core_conf_df[name_lower == q]
        if not matched.empty:
            return matched.iloc[0]["CORE Rank"]

        # 2. Exact abbreviation match — also check if abbrev appears as a
        #    token inside the source title (e.g. "...AAAI 2023...")
        matched = core_conf_df[abbrev_lower == q]
        if not matched.empty:
            return matched.iloc[0]["CORE Rank"]

        tokens = set(re.split(r"\W+", q))
        abbrev_in_title = core_conf_df[
            abbrev_lower.isin(tokens)
        ]
        if not abbrev_in_title.empty:
            return abbrev_in_title.iloc[0]["CORE Rank"]

    # 3. Fuzzy name match across all candidate names
    best_score = 0.0
    best_rank = ""
    for conf_name in names_to_match:
        q = conf_name.strip()
        if not q or q.lower() == "nan":
            continue
        for _, row in core_conf_df.iterrows():
            score = _fuzzy_score(q, row["Conference Name"])
            if score > best_score:
                best_score = score
                best_rank = row["CORE Rank"]

    if best_score >= fuzzy_threshold:
        return best_rank
    return ""


def check_in_core_journals(
    names_to_match: List[str],
    issn_to_match: List[str],
    core_journals_df: pd.DataFrame,
) -> str:
    """Check if any of the given names/ISSNs match a CORE journal. Returns rank or ''."""
    for name in names_to_match:
        matched = core_journals_df[
            core_journals_df["title"].str.strip().str.lower()
            == name.strip().lower()
        ]
        if not matched.empty:
            return matched.iloc[0]["rank"]

    for issn in issn_to_match:
        if not issn or issn == "":
            continue
        issn_clean = issn.strip().lower()
        matched = core_journals_df[
            (core_journals_df["ISSN1"].str.strip().str.lower() == issn_clean)
            | (core_journals_df["ISSN2"].str.strip().str.lower() == issn_clean)
            | (core_journals_df["ISSN3"].str.strip().str.lower() == issn_clean)
            | (core_journals_df["ISSN4"].str.strip().str.lower() == issn_clean)
        ]
        if not matched.empty:
            return matched.iloc[0]["rank"]

    return ""


def check_in_scimago_journals(
    names_to_match: List[str],
    issn_to_match: List[str],
    scimago_df: pd.DataFrame,
) -> Tuple[str, str]:
    """Check if any names/ISSNs match a SciMago journal. Returns (rank, areas) or ('', '').

    Fix 1 — ISSN normalization: SciMago stores ISSNs without dashes and as
    comma-separated pairs per row (e.g. '15424863, 00079235').  Papers from
    bib files usually carry dashes (e.g. '1542-4863').  Both sides are
    stripped of dashes/spaces before comparison so they always align.
    """
    for name in names_to_match:
        q = name.strip().lower()
        if not q or q == "nan":
            continue
        matched = scimago_df[
            scimago_df["Title"].str.strip().str.lower() == q
        ]
        if not matched.empty:
            return (
                matched.iloc[0]["SJR Best Quartile"],
                matched.iloc[0]["Areas"],
            )

    for issn in issn_to_match:
        if not issn or str(issn).strip().lower() in ("", "nan"):
            continue
        # Normalise: strip dashes/spaces, split on comma in case the bib
        # field contains multiple ISSNs (e.g. "1234-5678 2345-6789")
        for part in re.split(r"[,\s]+", issn):
            issn_norm = _normalize_issn(part)
            if not issn_norm:
                continue
            # SciMago Issn column may hold "15424863, 00079235" — split and compare
            matched = scimago_df[
                scimago_df["Issn"].apply(
                    lambda cell: issn_norm in [
                        _normalize_issn(x) for x in str(cell).split(",")
                    ]
                )
            ]
            if not matched.empty:
                return (
                    matched.iloc[0]["SJR Best Quartile"],
                    matched.iloc[0]["Areas"],
                )

    return "", ""


def check_predatory_publishers(
    publisher_to_check: str, pred_publishers_df: pd.DataFrame
) -> str:
    """Check if a publisher is in Beall's predatory list. Returns name or ''."""
    matched = pred_publishers_df[
        pred_publishers_df["publisher_name"].str.strip().str.lower()
        == publisher_to_check.strip().lower()
    ]
    if not matched.empty:
        return matched.iloc[0]["publisher_name"]
    return ""


def check_predatory_journals(
    journals_to_check: List[str], pred_journals_df: pd.DataFrame
) -> str:
    """Check if any journal names are in Beall's predatory list. Returns name or ''."""
    for j in journals_to_check:
        matched = pred_journals_df[
            pred_journals_df["journal_name"].str.strip().str.lower()
            == j.strip().lower()
        ]
        if not matched.empty:
            return matched.iloc[0]["journal_name"]
    return ""


def _build_scimago_index(scimago_df: pd.DataFrame) -> dict:
    """Pre-build O(1) lookup dicts for a SciMago DataFrame.

    Returns a dict with:
      'by_name': lowercase title → (quartile, areas)
      'by_issn': normalised ISSN  → (quartile, areas)
    """
    by_name = {}
    by_issn = {}
    for _, row in scimago_df.iterrows():
        quartile = row["SJR Best Quartile"]
        areas = row["Areas"]
        title = str(row["Title"]).strip().lower()
        if title and title != "nan":
            by_name[title] = (quartile, areas)
        for part in str(row["Issn"]).split(","):
            norm = _normalize_issn(part)
            if norm:
                by_issn[norm] = (quartile, areas)
    return {"by_name": by_name, "by_issn": by_issn}


def _build_core_conf_index(core_conf_df: pd.DataFrame) -> dict:
    """Pre-build O(1) lookup dicts for CORE conferences.

    Returns a dict with:
      'by_name': lowercase conference name → rank
      'by_abbrev': lowercase abbreviation  → rank
    """
    by_name = {}
    by_abbrev = {}
    for _, row in core_conf_df.iterrows():
        rank = row["CORE Rank"]
        name = str(row["Conference Name"]).strip().lower()
        abbrev = str(row["Abbreviation"]).strip().lower()
        if name and name != "nan":
            by_name[name] = rank
        if abbrev and abbrev != "nan":
            by_abbrev[abbrev] = rank
    return {"by_name": by_name, "by_abbrev": by_abbrev,
            "df": core_conf_df}  # df kept for fuzzy fallback


def _lookup_core_conf(names: List[str], index: dict, fuzzy_threshold: int) -> str:
    """O(1) name/abbreviation lookup with fuzzy fallback for conference papers."""
    by_name = index["by_name"]
    by_abbrev = index["by_abbrev"]

    for name in names:
        q = name.strip().lower()
        if not q or q == "nan":
            continue
        # 1. Exact name match
        if q in by_name:
            return by_name[q]
        # 2. Exact abbreviation match
        if q in by_abbrev:
            return by_abbrev[q]
        # 3. Abbreviation appears as a token inside the source title
        tokens = set(re.split(r"\W+", q))
        for abbrev, rank in by_abbrev.items():
            if abbrev in tokens:
                return rank

    # 4. Fuzzy fallback — only reached if no exact match found
    best_score = 0.0
    best_rank = ""
    core_conf_df = index["df"]
    for name in names:
        q = name.strip()
        if not q or q.lower() == "nan":
            continue
        for _, row in core_conf_df.iterrows():
            score = _fuzzy_score(q, row["Conference Name"])
            if score > best_score:
                best_score = score
                best_rank = row["CORE Rank"]

    return best_rank if best_score >= fuzzy_threshold else ""


def _lookup_scimago(names: List[str], issns: List[str], index: dict) -> Tuple[str, str]:
    """O(1) name/ISSN lookup against a pre-built SciMago index."""
    by_name = index["by_name"]
    by_issn = index["by_issn"]

    for name in names:
        q = name.strip().lower()
        if q and q != "nan" and q in by_name:
            return by_name[q]

    for issn in issns:
        for part in re.split(r"[,\s]+", str(issn)):
            norm = _normalize_issn(part)
            if norm and norm in by_issn:
                return by_issn[norm]

    return "", ""


def apply_rankings(
    df: pd.DataFrame,
    ranking_data: dict,
    enrich_venues: bool = False,
    openalex_email: Optional[str] = None,
    openalex_api_key: Optional[str] = None,
    core_fuzzy_threshold: int = 85,
    rank_strategy: str = "latest",
) -> pd.DataFrame:
    """Apply all ranking checks to a DataFrame, adding ranking flag columns.

    Parameters
    ----------
    df : pd.DataFrame
        Papers DataFrame (must have Source title, Journal, Book title,
        ISSN, ISBN, Publisher, Year, Entry Type).
    ranking_data : dict
        Output of load_ranking_data().
    enrich_venues : bool
        Fix 3 — when True, query OpenAlex for rows with no venue name
        before running ranking lookups (default False).
    openalex_email : str, optional
        Polite-pool email for OpenAlex (used when no API key is provided).
    openalex_api_key : str, optional
        OpenAlex API key for higher rate limits.
    core_fuzzy_threshold : int
        Fix 2 — fuzzy similarity threshold (0-100) for CORE conference
        name matching (default 85). Fuzzy matching only runs for papers
        whose Entry Type is in CONFERENCE_TYPES.
    rank_strategy : str
        How to select the SciMago snapshot year. "latest" (default) uses the
        most recent available snapshot (2024) for all papers. "publication_year"
        uses the snapshot closest to the paper's publication year, falling back
        to the latest when no exact match exists.

    Returns
    -------
    pd.DataFrame
        Copy of df with ranking columns added.
    """
    result = df.copy()

    if enrich_venues:
        result = enrich_missing_venues(
            result, api_key=openalex_api_key, email=openalex_email
        )

    # Initialize ranking columns
    for col in RANKING_COLUMNS:
        result[col] = ""

    core_conf = ranking_data["core_conf"]
    core_jour = ranking_data["core_jour"]
    scimago_2022 = ranking_data["scimago_2022"]
    scimago_2023 = ranking_data["scimago_2023"]
    scimago_2024 = ranking_data["scimago_2024"]
    pred_journals = ranking_data["pred_journals"]
    pred_publishers = ranking_data["pred_publishers"]

    # Pre-build indices for O(1) lookups (Fix 1 + Fix 2 performance)
    print("Building ranking lookup indices...")
    core_conf_index = _build_core_conf_index(core_conf)
    scimago_index_2022 = _build_scimago_index(scimago_2022)
    scimago_index_2023 = _build_scimago_index(scimago_2023)
    scimago_index_2024 = _build_scimago_index(scimago_2024)

    # Pre-build sets for predatory lookups
    pred_journal_names = set(
        pred_journals["journal_name"].str.strip().str.lower().dropna()
    )
    pred_publisher_names = set(
        pred_publishers["publisher_name"].str.strip().str.lower().dropna()
    )
    # Collect results in lists to assign in bulk (avoids slow .loc in loop)
    rows_core_conf = []
    rows_core_jour = []
    rows_pred_pub = []
    rows_pred_jour = []
    rows_scimago = {
        "is_scimago_q1_q2_2022": [], "scimago_2022_area": [],
        "is_scimago_q1_q2_2023": [], "scimago_2023_area": [],
        "is_scimago_q1_q2_2024": [], "scimago_2024_area": [],
    }

    print(f"Applying rankings to {len(result)} records...")
    for _, row in result.iterrows():
        year = str(row["Year"])
        source_title = str(row["Source title"])
        journal = str(row["Journal"])
        book_title = str(row["Book title"])
        issn = str(row["ISSN"]) if not pd.isna(row["ISSN"]) else ""
        isbn = str(row["ISBN"]) if not pd.isna(row["ISBN"]) else ""
        publisher = str(row["Publisher"])
        entry_type = str(row["Entry Type"])

        names = [source_title, book_title, journal]
        issns = [issn, isbn]

        # CORE conference — fuzzy only for conference-type entries
        fuzzy_thresh = core_fuzzy_threshold if entry_type in CONFERENCE_TYPES else 101
        rows_core_conf.append(_lookup_core_conf(names, core_conf_index, fuzzy_thresh))

        # CORE journal
        core_jour_rank = check_in_core_journals(names, issns, core_jour)
        rows_core_jour.append(core_jour_rank)

        # Predatory
        pub_lower = publisher.strip().lower()
        rows_pred_pub.append(publisher if pub_lower in pred_publisher_names else "")
        pred_j = next(
            (n for n in names if n.strip().lower() in pred_journal_names), ""
        )
        rows_pred_jour.append(pred_j)

        # SciMago — snapshot selection depends on rank_strategy
        for yr, idx_map, rank_col, area_col in [
            ("2022", scimago_index_2022, "is_scimago_q1_q2_2022", "scimago_2022_area"),
            ("2023", scimago_index_2023, "is_scimago_q1_q2_2023", "scimago_2023_area"),
            ("2024", scimago_index_2024, "is_scimago_q1_q2_2024", "scimago_2024_area"),
        ]:
            if rank_strategy == "latest":
                should_lookup = (yr == "2024")
            else:
                # publication_year: match snapshot to paper year, fall back to latest
                should_lookup = (year == yr) or (yr == "2024" and year not in ("2022", "2023"))
            if should_lookup:
                rank, area = _lookup_scimago(names, issns, idx_map)
                rows_scimago[rank_col].append(rank)
                rows_scimago[area_col].append(str(area))
            else:
                rows_scimago[rank_col].append("")
                rows_scimago[area_col].append("")

    # Bulk-assign results
    result["is_core_conference_A*_A_B"] = rows_core_conf
    result["is_core_journal_A*_A_B"] = rows_core_jour
    result["is_predatory_publisher_2025"] = rows_pred_pub
    result["is_predatory_journal_2025"] = rows_pred_jour
    for col, values in rows_scimago.items():
        result[col] = values

    return result


def old_filter_by_quality(
    df: pd.DataFrame,
    scimago_quartiles: list = None,
    core_ranks: list = None,
) -> pd.DataFrame:
    """Filter papers to keep only those published in quality venues.

    Parameters
    ----------
    df : pd.DataFrame
        Papers DataFrame with ranking columns (output of apply_rankings).
    scimago_quartiles : list
        Accepted SciMago quartiles, e.g. ["Q1", "Q2"]. Default: ["Q1", "Q2"].
    core_ranks : list
        Accepted CORE ranks, e.g. ["A*", "A", "B"]. Default: ["A*", "A", "B"].

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if scimago_quartiles is None:
        scimago_quartiles = ["Q1", "Q2"]
    if core_ranks is None:
        core_ranks = ["A*", "A", "B"]

    mask = (
        df["is_scimago_q1_q2_2024"].isin(scimago_quartiles)
        | df["is_scimago_q1_q2_2023"].isin(scimago_quartiles)
        | df["is_scimago_q1_q2_2022"].isin(scimago_quartiles)
        | df["is_core_conference_A*_A_B"].isin(core_ranks)
        | df["is_core_journal_A*_A_B"].isin(core_ranks)
    )

    filtered = df[mask].copy()
    excluded = df[~mask].copy()
    print(f"Papers after quality filtering: {len(filtered)}")
    print(
        f"Dropped {len(excluded)} papers not in ranked venues."
    )
    return filtered, excluded


def filter_by_quality(
    df: pd.DataFrame,
    scimago_quartiles: list = None,
    core_ranks: list = None,
) -> pd.DataFrame:
    """Filter papers to keep only those published in quality venues.

    Parameters
    ----------
    df : pd.DataFrame
        Papers DataFrame with ranking columns (output of apply_rankings).
    scimago_quartiles : list
        Accepted SciMago quartiles, e.g. ["Q1", "Q2"]. Default: ["Q1", "Q2"].
    core_ranks : list
        Accepted CORE ranks, e.g. ["A*", "A", "B"]. Default: ["A*", "A", "B"].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (filtered, excluded) DataFrames.
    """
    if scimago_quartiles is None:
        scimago_quartiles = ["Q1", "Q2"]
    if core_ranks is None:
        core_ranks = ["A*", "A", "B"]

    # --- Quality mask (same as old_filter_by_quality) ---
    quality_mask = (
        df["is_scimago_q1_q2_2024"].isin(scimago_quartiles)
        | df["is_scimago_q1_q2_2023"].isin(scimago_quartiles)
        | df["is_scimago_q1_q2_2022"].isin(scimago_quartiles)
        | df["is_core_conference_A*_A_B"].isin(core_ranks)
        | df["is_core_journal_A*_A_B"].isin(core_ranks)
    )

    filtered = df[quality_mask].copy()
    excluded = df[~quality_mask].copy()
    print(f"Papers after quality filtering: {len(filtered)}")
    print(f"Dropped {len(excluded)} papers not in ranked venues.")
    return filtered, excluded
