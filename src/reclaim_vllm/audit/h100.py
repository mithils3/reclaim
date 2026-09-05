"""Arithmetic checks and compute-band assignment for model H100 estimates."""

from __future__ import annotations

from typing import Any

H100_BANDS = ((0.0, 8.0, "0-8"), (8.0, 32.0, "8-32"), (32.0, 96.0, "32-96"), (96.0, 192.0, "96-192"))
OVER_CAP_BAND = ">192"
MISMATCH_TOLERANCE = 0.2

# The ladder above is the one source of truth for it. Everything downstream — the
# per-episode compute ceiling, the CLI help that documents it, the selection pool's
# eligible bands — derives from it here rather than restating the numbers.
BAND_MAX_HOURS = {label: high for _, high, label in H100_BANDS}


def band_max_hours(label: Any) -> float | None:
    """Upper edge of a band label in H100-h (``'96-192'`` -> 192.0).

    ``None`` for anything not on the ladder, including ``OVER_CAP_BAND``, which has
    no upper edge by construction.
    """
    if not isinstance(label, str):
        return None
    return BAND_MAX_HOURS.get(label.strip())


def band_labels(*, max_hours: float | None = None) -> tuple[str, ...]:
    """Ladder labels, optionally only those whose upper edge is within a cap."""
    return tuple(label for _, high, label in H100_BANDS
                 if max_hours is None or high <= max_hours)


def band_ladder_text() -> str:
    """The ladder as prose for CLI help: ``'0-8 -> 8h, 8-32 -> 32h, …'``."""
    return ", ".join(f"{label} -> {high:g}h" for _, high, label in H100_BANDS)


def h100_band(hours: Any) -> str | None:
    value = as_number(hours)
    if value is None or value < 0:
        return None
    for low, high, label in H100_BANDS:
        if low <= value <= high:
            return label
    return OVER_CAP_BAND


def recomputed_hours(estimate: dict[str, Any]) -> float | None:
    gpu_count = as_number(estimate.get("gpu_count"))
    wallclock = as_number(estimate.get("wallclock_hours"))
    multiplier = as_number(estimate.get("h100_equivalent_multiplier"))
    if gpu_count is None or wallclock is None or multiplier is None:
        return None
    if gpu_count <= 0 or wallclock < 0 or multiplier <= 0:
        return None
    return gpu_count * wallclock * multiplier


def arithmetic_mismatch(hours: Any, recomputed: float | None) -> bool | None:
    reported = as_number(hours)
    if reported is None or recomputed is None:
        return None
    largest = max(abs(reported), abs(recomputed))
    if largest == 0:
        return False
    return abs(reported - recomputed) / largest > MISMATCH_TOLERANCE


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
