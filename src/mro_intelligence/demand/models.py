"""Intermittent-demand models with probabilistic output.

Hand-rolled Croston, SBA, and TSB (standard published algorithms) rather than
the Nixtla path: the pipeline needs the (occurrence probability, demand size)
decomposition to compose quantiles, and hermetic deterministic unit tests need
full control of the smoothing updates. statsforecast/PyMC remain the refinement
path (see pyproject extras); they are not required for the deterministic MVP.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Model inputs are monthly demand vectors -- numpy arrays from the series
# builder, or plain lists in tests.
SeriesLike = np.ndarray | Sequence[float]

# Croston-family smoothing bounds: slow enough to stay stable on sparse
# series, fast enough to track real demand shifts.
MIN_SMOOTH_ALPHA = 0.05
MAX_SMOOTH_ALPHA = 0.30


def _alpha_for(n_updates: int) -> float:
    """Croston-style adaptive smoothing: 1/n of updates, bounded to [0.05, 0.3]."""
    if n_updates <= 0:
        return MAX_SMOOTH_ALPHA
    return float(np.clip(1.0 / n_updates, MIN_SMOOTH_ALPHA, MAX_SMOOTH_ALPHA))


def _validate(series: SeriesLike) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("demand series must be a non-empty 1-D array of non-negative values")
    return values


def croston(series: SeriesLike) -> tuple[float, float]:
    """Classic Croston: size and interval smoothed at demand occurrences.

    Returns (per-period occurrence probability, mean size when it occurs);
    the point forecast is their product, i.e. z / q in the classic form.
    """
    values = _validate(series)
    size: float | None = None
    interval = 1.0
    gap = 0
    updates = 0
    for demand in values:
        if demand > 0:
            if size is None:
                size = float(demand)
                interval = float(gap + 1)
            else:
                alpha = _alpha_for(updates)
                size += alpha * (float(demand) - size)
                interval += alpha * (float(gap + 1) - interval)
            updates += 1
            gap = 0
        else:
            gap += 1
    probability = 1.0 / interval if size is not None else 0.0
    return probability, size if size is not None else 0.0


def sba(series: SeriesLike) -> tuple[float, float]:
    """Syntetos-Boylan approximation: Croston with the bias correction factor.

    Croston's point forecast overestimates on intermittent series; SBA scales
    by (1 - alpha/2) with the final alpha the size smoothing used.
    """
    values = _validate(series)
    size: float | None = None
    interval = 1.0
    gap = 0
    updates = 0
    for demand in values:
        if demand > 0:
            if size is None:
                size = float(demand)
                interval = float(gap + 1)
            else:
                alpha = _alpha_for(updates)
                size += alpha * (float(demand) - size)
                interval += alpha * (float(gap + 1) - interval)
            updates += 1
            gap = 0
        else:
            gap += 1
    if size is None:
        return 0.0, 0.0
    alpha = _alpha_for(updates)
    probability = (1.0 - alpha / 2.0) / interval
    return probability, size


def tsb(series: SeriesLike, alpha_p: float = 0.1, alpha_z: float = 0.1) -> tuple[float, float]:
    """Teunter-Syntetos-Babai: updates occurrence probability every period.

    For lumpy series (long zero runs with occasional large demands) the
    per-period probability update reacts faster than Croston's interval form.
    """
    values = _validate(series)
    probability = float((values > 0).mean())
    size = float(values[values > 0].mean()) if np.any(values > 0) else 0.0
    for demand in values:
        occurred = 1.0 if demand > 0 else 0.0
        probability += alpha_p * (occurred - probability)
        if demand > 0:
            size += alpha_z * (float(demand) - size)
    return probability, size


def moving_average(series: SeriesLike, window: int = 12) -> float:
    """ERP-style flat-rate baseline: mean of the last `window` months.

    This is the "system baseline" the WMAPE gate compares against -- what a
    min/max or flat-forecast ERP method implies on the same history.
    """
    values = _validate(series)
    return float(values[-window:].mean())


def demand_class(series: SeriesLike) -> str:
    """Syntetos-Boylan quadrant from ADI and CV^2 of the train months.

    smooth (low ADI, low CV2) and erratic (low ADI, high CV2) behave close to
    a regular series; intermittent and lumpy are where Croston-family models
    earn their keep. Returns one of: smooth, erratic, intermittent, lumpy.
    """
    values = _validate(series)
    nonzero = values[values > 0]
    if len(nonzero) == 0:
        return "lumpy"  # all-zero: effectively never occurs
    adi = len(values) / len(nonzero)
    cv2 = float(nonzero.std(ddof=1) / nonzero.mean()) ** 2 if len(nonzero) > 1 else 0.0
    if adi <= 1.32 and cv2 <= 0.49:
        return "smooth"
    if adi <= 1.32:
        return "erratic"
    if cv2 <= 0.49:
        return "intermittent"
    return "lumpy"


def choose_model(series: SeriesLike) -> str:
    """Model selection rule: TSB on the zero-heavy quadrants, SBA elsewhere."""
    quadrant = demand_class(series)
    return "tsb" if quadrant in ("intermittent", "lumpy") else "sba"


def quantile_band(
    probability: float,
    size: float,
    nonzero_history: SeriesLike,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[float, float]:
    """Compose a monthly demand distribution from occurrence probability and size.

    Demand is zero with probability 1 - p; when it occurs its size draws from
    the SKU's recent nonzero-demand empirical distribution. Quantiles below the
    zero-mass are exactly 0 -- the spikiness of the long tail survives in the
    distribution instead of being averaged away into a single number.
    """
    history = np.asarray(nonzero_history, dtype=float)
    history = history[history > 0]
    p = float(np.clip(probability, 0.0, 1.0))
    sizes = history if len(history) else np.array([size], dtype=float)
    result: dict[float, float] = {}
    for tau in quantiles:
        if tau <= 1.0 - p or p <= 0.0:
            result[tau] = 0.0
            continue
        conditional = (tau - (1.0 - p)) / p
        result[tau] = float(np.quantile(sizes, float(np.clip(conditional, 0.0, 1.0))))
    return result
