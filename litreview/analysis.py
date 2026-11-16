"""Visualizations and bibliometric tables for quantitative analysis of SLR results."""

import re
from typing import Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud

BAR_COLOR = "#333333"
BAR_COLOR_SECONDARY = "#888888"

_AUTHOR_SPLIT_RE = re.compile(r"\s+and\s+|;")


def _style_axis(ax):
    """Remove top/right spines for an elegant minimal look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _shorten_author(name) -> str:
    """Normalize ``Last, First`` (or already-abbreviated ``Last, F.``) to ``Last, F.``."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    if "," not in name:
        return name
    last, _, first = name.partition(",")
    last = last.strip()
    first = first.strip()
    if not first:
        return last
    return f"{last}, {first[0].upper()}."


# BibTeX `@conference` is the deprecated alias of `@inproceedings`; both mean
# "paper in conference proceedings" — so they aggregate to Conferences, not Journals.
_ENTRY_TYPE_TO_VENUE = {
    "article": "Journals",
    "IEEE Journals": "Journals",
    "conference": "Conferences",
    "inproceedings": "Conferences",
}


def _draw_year_distribution(ax, df: pd.DataFrame):
    counts = df["Year"].dropna().astype(int).value_counts().sort_index()
    bars = ax.bar(
        counts.index.astype(str), counts.values, color=BAR_COLOR, width=0.5
    )
    ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Number of Publications", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(ax)


def _draw_entry_type_pie(ax, df: pd.DataFrame):
    types = df["Entry Type"].dropna().astype(str).str.strip()
    venues = types.map(_ENTRY_TYPE_TO_VENUE).dropna()
    counts = venues.value_counts()
    total = int(counts.sum())
    palette = [BAR_COLOR, BAR_COLOR_SECONDARY][: len(counts)]
    _, _, pct_texts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=lambda p: f"{int(round(p * total / 100))}",
        colors=palette,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 8},
        startangle=90,
    )
    for t in pct_texts:
        t.set_color("white")
    ax.set_aspect("equal")


def _draw_hbar(ax, counts: pd.Series, xlabel: str, bar_height: float = 0.6):
    sns.barplot(
        y=counts.index,
        x=counts.values,
        color=BAR_COLOR,
        ax=ax,
        width=bar_height,
    )
    if ax.containers:
        ax.bar_label(ax.containers[0], fontsize=7, padding=2)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(ax)


def plot_year_distribution(df: pd.DataFrame, figsize=(4, 2.5)):
    """Bar chart of publication year distribution."""
    _, ax = plt.subplots(figsize=figsize)
    _draw_year_distribution(ax, df)
    plt.tight_layout()
    plt.show()


def plot_entry_type_distribution(df: pd.DataFrame, figsize=(3, 2.8)):
    """Pie chart of venue categories (Journals vs Conferences)."""
    _, ax = plt.subplots(figsize=figsize)
    _draw_entry_type_pie(ax, df)
    plt.tight_layout()
    plt.show()


def plot_year_and_venue(df: pd.DataFrame, figsize=(6.5, 2.5)):
    """Year distribution and venue pie side by side — paper-ready combined figure."""
    _, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]}
    )
    _draw_year_distribution(ax_bar, df)
    _draw_entry_type_pie(ax_pie, df)
    plt.tight_layout()
    plt.show()


def _pick_display_spelling(forms: pd.Series) -> str:
    """Pick the spelling with the lowest uppercase ratio (avoids WoS all-caps forms)."""
    forms = forms.dropna().unique()
    if len(forms) == 1:
        return forms[0]

    def upper_ratio(s):
        letters = [c for c in s if c.isalpha()]
        return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0

    return min(forms, key=upper_ratio)


def plot_top_sources(df: pd.DataFrame, n: int = 10, figsize=(9, 3)):
    """Horizontal bar chart of top venues (coalesces Source title / Journal / Book title).

    Groups venues case-insensitively so WoS all-caps forms aren't counted separately from
    the Scopus/ACM title-case forms of the same journal.
    """
    venues = (
        df["Source title"]
        .fillna(df.get("Journal"))
        .fillna(df.get("Book title"))
        .dropna()
        .astype(str)
        .str.strip()
    )
    venues_lower = venues.str.lower()
    counts = venues_lower.value_counts().head(n)
    display = {
        low: _pick_display_spelling(venues[venues_lower == low])
        for low in counts.index
    }
    counts.index = [display[i] for i in counts.index]

    _, ax = plt.subplots(figsize=figsize)
    _draw_hbar(ax, counts, xlabel="Number of Publications")
    plt.tight_layout()
    plt.show()


def plot_top_authors(df: pd.DataFrame, n: int = 10, figsize=(4.5, 3)):
    """Horizontal bar chart of top authors across the whole author pool, formatted ``Last, F.``."""
    authors_split = (
        df["Authors"]
        .dropna()
        .astype(str)
        .str.split(_AUTHOR_SPLIT_RE)
        .explode()
        .str.strip()
    )
    authors_split = authors_split[authors_split.str.len() > 0].map(_shorten_author)
    top_authors = authors_split.value_counts().head(n)

    _, ax = plt.subplots(figsize=figsize)
    _draw_hbar(ax, top_authors, xlabel="Number of Publications")
    plt.tight_layout()
    plt.show()


def plot_top_first_authors(df: pd.DataFrame, n: int = 10, figsize=(4.5, 3)):
    """Horizontal bar chart of top first authors only, formatted ``Last, F.``."""
    first_authors = (
        df["Authors"]
        .dropna()
        .astype(str)
        .str.split(_AUTHOR_SPLIT_RE, n=1)
        .str[0]
        .str.strip()
        .map(_shorten_author)
    )
    first_authors = first_authors[first_authors.str.len() > 0]
    top_first = first_authors.value_counts().head(n)

    _, ax = plt.subplots(figsize=figsize)
    _draw_hbar(ax, top_first, xlabel="Number of First-Author Publications")
    plt.tight_layout()
    plt.show()


def plot_top_keywords(df: pd.DataFrame, n: int = 20, figsize=(5, 4)):
    """Horizontal bar chart of top author keywords."""
    keywords_split = (
        df["Author Keywords"]
        .dropna()
        .str.lower()
        .str.split(r";|,")
        .explode()
        .str.strip()
    )
    top_keywords = keywords_split.value_counts().head(n)

    _, ax = plt.subplots(figsize=figsize)
    _draw_hbar(ax, top_keywords, xlabel="Count")
    plt.tight_layout()
    plt.show()


def plot_keyword_wordcloud(df: pd.DataFrame, figsize=(7, 3.5), colormap: str = "Blues"):
    """Word cloud of author keywords. ``colormap`` accepts any matplotlib colormap name."""
    keywords_split = (
        df["Author Keywords"]
        .dropna()
        .str.lower()
        .str.split(r";|,")
        .explode()
        .str.strip()
    )
    all_keywords = " ".join(keywords_split.dropna())
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
    ).generate(all_keywords)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Bibliometric tables (curated quantitative sheet)
# ---------------------------------------------------------------------------

OutputFormat = Literal["dataframe", "markdown", "latex"]
KeywordLevel = Literal["hierarchy", "specific", "both"]

DEFAULT_TOTAL = 50  # included-studies corpus size; missingness reported against this


def _caption(label: str, n: int, total: int) -> str:
    missing = total - n
    pct = 100 * missing / total if total else 0
    return f"{label} — n = {n} of {total} ({pct:.0f}% missing)"


def _render(
    df: pd.DataFrame, caption: str, output: OutputFormat
) -> Union[pd.DataFrame, str]:
    if output == "dataframe":
        print(caption)
        return df
    if output == "markdown":
        return f"**{caption}**\n\n{df.to_markdown(index=False)}"
    if output == "latex":
        return df.to_latex(index=False, caption=caption)
    raise ValueError(f"Unknown output format: {output!r}")


def _value_counts_table(
    series: pd.Series,
    *,
    label_col: str,
    top_n: Union[int, None],
) -> pd.DataFrame:
    """Counts + % table. If ``top_n`` is set and exceeded, append 'Other (k)'."""
    s = series.dropna()
    total = len(s)
    counts = s.value_counts()
    pct = (counts / total * 100).round(1) if total else counts * 0
    table = pd.DataFrame(
        {label_col: counts.index, "Count": counts.values, "%": pct.values}
    )
    if top_n is not None and len(table) > top_n:
        head = table.head(top_n).copy()
        tail = table.iloc[top_n:]
        other_count = int(tail["Count"].sum())
        other_pct = round(other_count / total * 100, 1) if total else 0.0
        head.loc[len(head)] = [f"Other ({len(tail)})", other_count, other_pct]
        table = head.reset_index(drop=True)
    return table


def missing_values_overview(
    df: pd.DataFrame,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    """Per-column non-null count and % completeness against the corpus total."""
    n_total = total or len(df)
    rows = [
        {
            "Column": col,
            "Non-null": int(df[col].notna().sum()),
            "Missing": n_total - int(df[col].notna().sum()),
            "% complete": round(100 * int(df[col].notna().sum()) / n_total, 1),
        }
        for col in df.columns
    ]
    return _render(
        pd.DataFrame(rows),
        _caption("Missingness overview", n_total, n_total),
        output,
    )


def country_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    n_total = total or len(df)
    n = int(df["Country"].notna().sum())
    table = _value_counts_table(df["Country"], label_col="Country", top_n=top_n)
    return _render(table, _caption("Country distribution", n, n_total), output)


def continent_distribution(
    df: pd.DataFrame,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    n_total = total or len(df)
    n = int(df["Continent"].notna().sum())
    table = _value_counts_table(df["Continent"], label_col="Continent", top_n=None)
    return _render(table, _caption("Continent distribution", n, n_total), output)


def area_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    """Counts by Area; compound entries ('A; B') are split and counted multi-label."""
    n_total = total or len(df)
    n_papers = int(df["Area"].notna().sum())
    areas = [
        part.strip()
        for value in df["Area"].dropna()
        for part in str(value).split(";")
        if part.strip()
    ]
    table = _value_counts_table(
        pd.Series(areas, name="Area"), label_col="Area", top_n=top_n
    )
    return _render(
        table,
        _caption("Area distribution (multi-label; % over mentions)", n_papers, n_total),
        output,
    )


def keyword_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    keyword_level: KeywordLevel = "both",
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str, dict]:
    """ACM-style keyword counts.

    - ``"hierarchy"``: joins Keyword 1 → 2 → 3 with ' → ' per paper.
    - ``"specific"``: takes the deepest non-null among K1/K2/K3 per paper.
    - ``"both"``: returns a dict ``{"hierarchy": ..., "specific": ...}``.
    """
    n_total = total or len(df)
    cols = [c for c in ("Keyword 1", "Keyword 2", "Keyword 3") if c in df.columns]

    def _parts(row):
        return [
            str(row[c]).strip()
            for c in cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]

    def _hierarchy_series() -> pd.Series:
        items = [
            " → ".join(p) for p in (_parts(row) for _, row in df.iterrows()) if p
        ]
        return pd.Series(items, name="Keyword hierarchy")

    def _specific_series() -> pd.Series:
        items = [p[-1] for p in (_parts(row) for _, row in df.iterrows()) if p]
        return pd.Series(items, name="Keyword (most specific)")

    def _build(series: pd.Series, label: str):
        n = len(series)
        table = _value_counts_table(series, label_col=series.name, top_n=top_n)
        return _render(table, _caption(label, n, n_total), output)

    if keyword_level == "hierarchy":
        return _build(_hierarchy_series(), "Keyword hierarchy distribution")
    if keyword_level == "specific":
        return _build(_specific_series(), "Most-specific keyword distribution")
    if keyword_level == "both":
        return {
            "hierarchy": _build(
                _hierarchy_series(), "Keyword hierarchy distribution"
            ),
            "specific": _build(
                _specific_series(), "Most-specific keyword distribution"
            ),
        }
    raise ValueError(f"Unknown keyword_level: {keyword_level!r}")


def venue_quality_summary(
    df: pd.DataFrame,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    """Descriptive stats (median, IQR, min, max, missingness) for SJR and H-Index."""
    n_total = total or len(df)
    rows = []
    for col in ("SJR (2024)", "H-Index"):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s):
            rows.append(
                {
                    "Metric": col,
                    "n": len(s),
                    "Median": round(s.median(), 3),
                    "IQR": f"{round(s.quantile(0.25), 3)}–{round(s.quantile(0.75), 3)}",
                    "Min": round(s.min(), 3),
                    "Max": round(s.max(), 3),
                    "Missing": n_total - len(s),
                    "% missing": round(100 * (n_total - len(s)) / n_total, 1),
                }
            )
        else:
            rows.append(
                {
                    "Metric": col,
                    "n": 0,
                    "Median": None,
                    "IQR": None,
                    "Min": None,
                    "Max": None,
                    "Missing": n_total,
                    "% missing": 100.0,
                }
            )
    return _render(
        pd.DataFrame(rows),
        _caption("Venue quality summary", n_total, n_total),
        output,
    )


_CORE_GRADE_ORDER = ["A*", "A", "B", "C"]


def core_ranking_distribution(
    df: pd.DataFrame,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    n_total = total or len(df)
    n = int(df["CORE Ranking"].notna().sum())
    table = _value_counts_table(
        df["CORE Ranking"], label_col="CORE Ranking", top_n=None
    )
    table["_order"] = table["CORE Ranking"].map(
        lambda g: _CORE_GRADE_ORDER.index(g) if g in _CORE_GRADE_ORDER else 99
    )
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return _render(
        table, _caption("CORE ranking distribution", n, n_total), output
    )


def top_venues_by_sjr(
    df: pd.DataFrame,
    top_n: int = 10,
    include_h_index: bool = True,
    order_by_h_index: bool = False,
    order_by_num_papers: bool = True,
    total: Union[int, None] = None,
    output: OutputFormat = "dataframe",
) -> Union[pd.DataFrame, str]:
    """Top venues ranked by paper count (default), SJR (2024), or H-Index.

    Ordering precedence: ``order_by_num_papers`` > ``order_by_h_index`` > SJR.

    - ``order_by_num_papers=True`` (default): rank by number of papers in the
      corpus; all venues kept, SJR/H-Index may be NaN. Ties broken by SJR desc.
    - ``order_by_h_index=True`` (with ``order_by_num_papers=False``): rank by
      H-Index; venues missing H-Index are dropped.
    - both False: rank by SJR; venues missing SJR are dropped.
    - ``include_h_index=False``: omit the H-Index column from the output.
    """
    n_total = total or len(df)
    if order_by_num_papers:
        subset = df
        sort_cols = ["Papers", "SJR (2024)"]
        label = "Top venues by number of papers"
    elif order_by_h_index:
        subset = df.dropna(subset=["H-Index"])
        sort_cols = ["H-Index", "Papers"]
        label = "Top venues by H-Index"
    else:
        subset = df.dropna(subset=["SJR (2024)"])
        sort_cols = ["SJR (2024)", "Papers"]
        label = "Top venues by SJR (2024)"
    n = len(subset)

    grouped = (
        subset.groupby("Venue", dropna=True)
        .agg(
            Papers=("Title", "size"),
            SJR=("SJR (2024)", "first"),
            H=("H-Index", "first"),
        )
        .reset_index()
        .rename(columns={"SJR": "SJR (2024)", "H": "H-Index"})
        .sort_values(sort_cols, ascending=False, na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )
    if not include_h_index:
        grouped = grouped.drop(columns="H-Index")
    return _render(grouped, _caption(label, n, n_total), output)
