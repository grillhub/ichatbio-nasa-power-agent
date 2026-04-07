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


async def sanitize_locations(locations: list[dict[str, Any]], process: Any) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []

    for idx, loc in enumerate(locations):
        try:
            lat = float(loc.get("decimalLatitude"))
            lon = float(loc.get("decimalLongitude"))
        except (TypeError, ValueError):
            await process.log(
                f"Location {idx + 1}: skipped — latitude or longitude is missing or not a valid number."
            )
            continue

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            await process.log(
                f"Location {idx + 1}: skipped — latitude must be between -90 and 90, longitude between -180 and 180."
            )
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
            await process.log(
                f"Location {idx + 1}: skipped — start date is only a year ({raw_start}). Use a full date, for example 2007-06-19."
            )
            continue
        if raw_end is not None and end_norm is None and len(str(raw_end).strip()) == 4 and str(raw_end).strip().isdigit():
            await process.log(
                f"Location {idx + 1}: skipped — end date is only a year ({raw_end}). Use a full date, for example 2007-06-19."
            )
            continue

        if not start_norm or not end_norm:
            await process.log(
                f"Location {idx + 1}: skipped — start or end date is missing or not in a supported format (start: {raw_start}, end: {raw_end})."
            )
            continue

        try:
            start_year = int(start_norm[:4])
            end_year = int(end_norm[:4])
        except ValueError:
            await process.log(
                f"Location {idx + 1}: skipped — could not read the year from the start or end date."
            )
            continue

        if start_year < 1981 or end_year < 1981:
            await process.log(
                f"Location {idx + 1}: skipped — data is only available from 1981 onward (your range includes an earlier year)."
            )
            continue

        try:
            validate_date(start_norm)
            validate_date(end_norm)
        except ValueError as e:
            await process.log(
                f"Location {idx + 1}: skipped — {e}"
            )
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

    if len(sanitized) != len(locations):
        skipped = len(locations) - len(sanitized)
        await process.log(
            f"Validation summary: {len(sanitized)} location(s) ready to use; {skipped} skipped ({len(locations)} total)."
        )

    return sanitized