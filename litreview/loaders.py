"""Functions for loading paper data from various database export formats."""

import os

import bibtexparser
import numpy as np
import pandas as pd


# Standard column schema used across all databases
STANDARD_COLUMNS = [
    "Title",
    "Authors",
    "Year",
    "Abstract",
    "DOI",
    "ISSN",
    "ISBN",
    "Source title",
    "Journal",
    "Book title",
    "Publisher",
    "Series",
    "Author Keywords",
    "Keywords",
    "Entry Type",
    "Database",
]

# IEEE Xplore columns that are not needed in the merged dataset
IEEE_COLS_TO_DROP = [
    "Date Added To Xplore",
    "Volume",
    "Issue",
    "Start Page",
    "End Page",
    "Funding Information",
    "PDF Link",
    "IEEE Terms",
    "Mesh_Terms",
    "Article Citation Count",
    "Patent Citation Count",
    "Reference Count",
    "License",
    "Online Date",
    "Issue Date",
    "Meeting Date",
]

# Mapping from IEEE Xplore column names (after dropping) to standard names
IEEE_COLUMN_NAMES = [
    "Title",
    "Authors",
    "Author Affiliations",
    "Source title",
    "Year",
    "Abstract",
    "ISSN",
    "ISBN",
    "DOI",
    "Author Keywords",
    "Publisher",
    "Entry Type",
    "Database",
]


# Manual cleanup applied to the curated bibliometric sheet.
QUANT_AREA_NORMALIZATION = {
    "Human-computer interaction (HCI)": "HCI",
}


def load_quantitative_sheet(
    xlsx_path: str,
    sheet_name: str = "additional_data_for_quantitativ",
) -> pd.DataFrame:
    """Load the curated bibliometric sheet for the included studies.

    Strips whitespace from string columns, applies known area surface
    normalizations, drops fully-empty rows/columns from the sheet's reserved
    extent, and coerces SJR / H-Index to numeric.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    if "Area" in df.columns:
        df["Area"] = df["Area"].map(
            lambda v: QUANT_AREA_NORMALIZATION.get(v, v) if isinstance(v, str) else v
        )

    for col in ("SJR (2024)", "H-Index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def bib_entries_to_df(entries: list) -> pd.DataFrame:
    """Convert bibtexparser entries to a DataFrame with standard columns."""
    records = []
    for entry in entries:
        record = {
            "Title": entry.get("title", "").strip(),
            "Authors": entry.get("author", "").strip(),
            "Year": (
                int(entry.get("year", "0")[:4]) if "year" in entry else None
            ),
            "Abstract": entry.get("abstract", "").strip(),
            "Source": entry.get("ID", ""),
            "DOI": entry.get("doi", ""),
            "ISSN": entry.get("issn", ""),
            "ISBN": entry.get("isbn", ""),
            "Source title": entry.get("journal", ""),
            "Journal": entry.get("journal", ""),
            "Book title": entry.get("booktitle", ""),
            "Publisher": entry.get("publisher", ""),
            "Series": entry.get("series", ""),
            "Author Keywords": entry.get("keywords", ""),
            "Keywords": entry.get("keywords", ""),
            "Entry Type": entry.get("ENTRYTYPE", ""),
        }
        records.append(record)
    return pd.DataFrame(records)


def load_bib_file(path: str, database_name: str) -> pd.DataFrame:
    """Load a .bib file and return a DataFrame with a Database column."""
    with open(path, "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)
    df = bib_entries_to_df(bib_database.entries)
    df["Database"] = database_name
    print(f"{database_name} finished! ({len(df)} records)")
    return df


def load_ieee_csv(path: str) -> pd.DataFrame:
    """Load an IEEE Xplore CSV, rename columns to standard schema."""
    df = pd.read_csv(path)
    df["Database"] = "IEEE Xplore"

    # Drop columns that exist in the dataframe
    existing_cols_to_drop = [c for c in IEEE_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    df.columns = IEEE_COLUMN_NAMES

    # Add missing standard columns
    df["Journal"] = np.nan
    df["Book title"] = np.nan
    df["Series"] = np.nan
    df["Keywords"] = np.nan

    # Drop non-standard columns
    df = df.drop(columns=["Author Affiliations"], errors="ignore")

    print(f"IEEE Xplore finished! ({len(df)} records)")
    return df


def load_all_databases(
    file_paths: dict,
    use_cached_csv: bool = False,
    csv_dir: str = None,
) -> pd.DataFrame:
    """Load all databases and merge into a single DataFrame.

    Parameters
    ----------
    file_paths : dict
        Mapping of database name to file path. Expected keys:
        "ACM Digital Library", "Scopus", "WoS", "IEEE Xplore".
    use_cached_csv : bool
        If True, load from pre-exported CSVs in csv_dir instead of raw files.
    csv_dir : str
        Directory containing cached CSV exports.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all records from all databases.
    """
    if use_cached_csv and csv_dir:
        bib_database_acm = pd.read_csv(
            os.path.join(csv_dir, "acm_digital_library.csv")
        )
        bib_database_scopus = pd.read_csv(
            os.path.join(csv_dir, "scopus.csv")
        )
        bib_database_wos = pd.read_csv(os.path.join(csv_dir, "wos.csv"))
        ieeexplore_df = pd.read_csv(
            os.path.join(csv_dir, "ieee_explore.csv")
        )
    else:
        # Load BibTeX databases
        bib_database_acm = load_bib_file(
            file_paths["ACM Digital Library"], "ACM Digital Library"
        )
        bib_database_scopus = load_bib_file(
            file_paths["Scopus"], "Scopus"
        )
        bib_database_wos = load_bib_file(file_paths["WoS"], "WoS")

        # Load IEEE (supports both .bib and .csv)
        ieee_path = file_paths["IEEE Xplore"]
        if ieee_path.endswith(".bib"):
            ieeexplore_df = load_bib_file(ieee_path, "IEEE Xplore")
        else:
            ieeexplore_df = load_ieee_csv(ieee_path)

        # Optionally export to CSV for caching
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            bib_database_acm.to_csv(
                os.path.join(csv_dir, "acm_digital_library.csv"), index=False
            )
            bib_database_scopus.to_csv(
                os.path.join(csv_dir, "scopus.csv"), index=False
            )
            bib_database_wos.to_csv(
                os.path.join(csv_dir, "wos.csv"), index=False
            )
            ieeexplore_df.to_csv(
                os.path.join(csv_dir, "ieee_explore.csv"), index=False
            )
            print("All results exported to CSV!")

    # Merge BibTeX-sourced databases first
    bib_dfs = [bib_database_acm, bib_database_scopus, bib_database_wos]
    merged = pd.concat(bib_dfs, ignore_index=True)

    # Drop the Source column from bib data (not present in IEEE)
    merged = merged.drop(columns=["Source"], errors="ignore")

    # Align IEEE columns to match merged, then concatenate
    common_cols = [c for c in merged.columns if c in ieeexplore_df.columns]
    ieeexplore_df = ieeexplore_df[common_cols]

    all_results = pd.concat([merged, ieeexplore_df], ignore_index=True)
    print(f"\nTotal records loaded: {len(all_results)}")
    return all_results
