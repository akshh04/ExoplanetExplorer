"""Small, dependency-free checks for the NASA archive response contract."""

REQUIRED_ARCHIVE_COLUMNS = frozenset(
    {
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "pl_orbsmax",
        "pl_eqt",
        "st_teff",
        "st_lum",
        "sy_dist",
    }
)


def validate_archive_columns(columns) -> None:
    """Raise a clear error when NASA changes or truncates the selected schema."""
    missing = sorted(REQUIRED_ARCHIVE_COLUMNS.difference(columns))
    if missing:
        raise ValueError(
            "NASA archive response is missing required columns: " + ", ".join(missing)
        )
