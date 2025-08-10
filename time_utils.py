from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any, Iterable

import pandas as pd

ISO_Z_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def to_utc_datetime(value: Any) -> Optional[datetime]:
    """
    Parse various timestamp representations to a timezone-aware UTC datetime.
    Returns None if the value can't be parsed.
    Accepted inputs: datetime, str (ISO8601 or common), int/float (unix seconds).
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            # pandas handles many formats; use utc=True to unify tz
            dt = pd.to_datetime(value, utc=True, errors="coerce").to_pydatetime()
        except Exception:
            return None

    if dt is pd.NaT or dt is None:
        return None

    # Ensure timezone-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def to_iso_utc_z(value: Any) -> Optional[str]:
    """
    Convert input value to ISO 8601 string in UTC with trailing 'Z'.
    Example: 2025-08-10T00:00:00.000000Z
    Returns None if parsing fails.
    """
    dt = to_utc_datetime(value)
    if dt is None:
        return None
    # Use isoformat then enforce Z, or use strftime for stable microsecond formatting
    try:
        # isoformat keeps microseconds; replace timezone with Z
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return dt.strftime(ISO_Z_FORMAT)


def normalize_series_to_iso_utc(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series of timestamps to ISO8601 UTC Z strings.
    Invalid entries become None. Returns object dtype series.
    """
    def _convert(x: Any) -> Optional[str]:
        if pd.isna(x):
            return None
        return to_iso_utc_z(x)

    return series.apply(_convert).astype("object")


def now_iso_utc_z() -> str:
    """Current UTC time as ISO 8601 Z string."""
    return to_iso_utc_z(datetime.now(timezone.utc)) or datetime.now(timezone.utc).strftime(ISO_Z_FORMAT)
