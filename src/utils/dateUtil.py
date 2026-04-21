import re
from datetime import UTC, datetime, timedelta
from typing import Any


def validate_date(date_str: str) -> None:

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")

    if dt.date() > datetime.now(UTC).date():
        raise ValueError(f"Date '{date_str}' is in the future. Please provide a date up to today.")


def _epoch_to_ymd(value: int | float) -> str | None:
    """Convert Unix epoch (seconds or milliseconds) to YYYY-MM-DD in UTC."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        # Heuristic: seconds are typically 1e8–2e9; ms are >= 1e10 (JS-style) or > 1e11.
        # Values > 1e11 cannot be seconds (would be far past year 2100 as Unix seconds).
        if value > 1e11:
            secs = value / 1000.0
        elif value >= 1e10:
            secs = value / 1000.0
        elif value > 1e9:
            secs = float(value)
        else:
            return None
        dt = datetime.fromtimestamp(secs, tz=UTC)
        return dt.strftime("%Y-%m-%d")
    except (OSError, ValueError, OverflowError):
        return None


def _normalize_to_ymd(date_str: Any) -> str | None:
    if date_str is None:
        return None

    # Integer/float epoch (seconds or milliseconds), not bool (bool is a subclass of int).
    if isinstance(date_str, (int, float)) and not isinstance(date_str, bool):
        ymd = _epoch_to_ymd(date_str)
        if ymd is not None:
            return ymd

    s = str(date_str).strip()
    if not s:
        return None

    # String that is only digits may be a Unix epoch (seconds or ms), e.g. "1167782400000".
    if s.isdigit() and len(s) >= 9:
        ymd = _epoch_to_ymd(int(s))
        if ymd is not None:
            return ymd

    # Year-only (e.g. "2007") is not usable for NASA POWER daily queries.
    if len(s) == 4 and s.isdigit():
        return None

    # If it already contains YYYY-MM-DD anywhere, use that.
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)

    # Convert YYYY/MM/DD -> YYYY-MM-DD
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Common US style (MM/DD/YYYY)
    for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def sanitize_locations(
    locations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sanitized: list[dict[str, Any]] = []
    error_count: dict[str, int] = {
        "lat_long_valid": 0,
        "lat_long_range": 0,
        "start_date_year_only": 0,
        "end_date_year_only": 0,
        "dates_missing_or_unsupported": 0,
        "year_parse_failed": 0,
        "year_before_1981": 0,
        "date_validation_failed": 0,
    }

    for loc in locations:
        try:
            lat = float(loc.get("decimalLatitude"))
            lon = float(loc.get("decimalLongitude"))
        except (TypeError, ValueError):
            error_count["lat_long_valid"] += 1
            continue

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            error_count["lat_long_range"] += 1
            continue

        # --- Date normalization ---
        raw_start = loc.get("startDate")
        raw_end = loc.get("endDate")

        # If a single field contains a range like "YYYY-MM-DD/YYYY-MM-DD", split it.
        for raw in (raw_start, raw_end):
            if raw is None:
                continue
            s = str(raw).strip()
            if "/" in s:
                parts = [p.strip() for p in s.split("/") if p.strip()]
                if len(parts) >= 2:
                    raw_start = parts[0]
                    raw_end = parts[1]
                    break

        start_norm = _normalize_to_ymd(raw_start)
        end_norm = _normalize_to_ymd(raw_end)

        if raw_start is not None and start_norm is None and len(str(raw_start).strip()) == 4 and str(raw_start).strip().isdigit():
            error_count["start_date_year_only"] += 1
            continue
        if raw_end is not None and end_norm is None and len(str(raw_end).strip()) == 4 and str(raw_end).strip().isdigit():
            error_count["end_date_year_only"] += 1
            continue

        if not start_norm or not end_norm:
            error_count["dates_missing_or_unsupported"] += 1
            continue

        try:
            start_year = int(start_norm[:4])
            end_year = int(end_norm[:4])
        except ValueError:
            error_count["year_parse_failed"] += 1
            continue

        if start_year < 1981 or end_year < 1981:
            error_count["year_before_1981"] += 1
            continue

        try:
            validate_date(start_norm)
            validate_date(end_norm)
        except ValueError:
            error_count["date_validation_failed"] += 1
            continue

        loc2 = dict(loc)
        loc2["decimalLatitude"] = lat
        loc2["decimalLongitude"] = lon
        loc2["startDate"] = start_norm
        loc2["endDate"] = end_norm
        if start_norm != loc.get("startDate"):
            loc2["originalStartDate"] = loc.get("startDate")
        if end_norm != loc.get("endDate"):
            loc2["originalEndDate"] = loc.get("endDate")

        sanitized.append(loc2)

    return sanitized, error_count