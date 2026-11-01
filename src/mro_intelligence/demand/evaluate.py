"""Held-out WMAPE evaluation of the demand forecast vs the ERP baseline.

The S2 KPI gate: long-tail WMAPE 15-25% better than the ERP method on
held-out months. The ERP baseline is the AXON plan's "ERP as-is" belief --
a flat trailing-mean rate with no seasonality. The gate is scored on the
Z-quadrant (the intermittent long tail the module targets); the overall
catalog WMAPE is reported alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WMAPE_GATE = 0.15


@dataclass(frozen=True)
class WmapeResult:
    """Pooled WMAPE comparison on the held-out horizon, overall and long tail."""

    wmape_baseline: float
    wmape_model: float
    improvement: float
    wmape_baseline_longtail: float
    wmape_model_longtail: float
    improvement_longtail: float
    n_sku_months: int

    def passes_gate(self, gate: float = WMAPE_GATE) -> bool:
        return self.n_sku_months > 0 and self.improvement_longtail >= gate


def _wmape(forecast: pd.Series, actual: pd.Series) -> float:
    """Pooled WMAPE: sum|a-f| / sum|a| -- volume-weighted, scale-free."""
    denom = actual.abs().sum()
    if denom <= 0:
        return 0.0
    return float((actual - forecast).abs().sum() / denom)


def evaluate_wmape(
    eval_df: pd.DataFrame,
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    segmentation: pd.DataFrame,
) -> WmapeResult:
    """Score forecaster frames on the held-out (sku, month) rows.

    Both forecaster frames carry (sku, month, forecast); the evaluator joins
    them to the actuals and to the XYZ classes so the long-tail (Z-quadrant)
    gate and the overall figure come from one shared denominator set.
    """
    joined = eval_df.rename(columns={"qty": "actual"})[["sku", "month", "actual"]]
    joined = joined.merge(model, on=["sku", "month"], how="left")
    joined = joined.merge(baseline, on=["sku", "month"], how="left", suffixes=("", "_base"))
    joined = joined.merge(segmentation[["sku", "xyz_class"]], on="sku", how="left")
    for col in ("forecast", "forecast_base"):
        joined[col] = joined[col].fillna(0.0)

    longtail = joined[joined["xyz_class"] == "Z"]
    wmape_baseline = _wmape(joined["forecast_base"], joined["actual"])
    wmape_model = _wmape(joined["forecast"], joined["actual"])
    wmape_baseline_lt = _wmape(longtail["forecast_base"], longtail["actual"])
    wmape_model_lt = _wmape(longtail["forecast"], longtail["actual"])
    improvement = (wmape_baseline - wmape_model) / wmape_baseline if wmape_baseline > 0 else 0.0
    improvement_lt = (
        (wmape_baseline_lt - wmape_model_lt) / wmape_baseline_lt if wmape_baseline_lt > 0 else 0.0
    )
    return WmapeResult(
        wmape_baseline=wmape_baseline,
        wmape_model=wmape_model,
        improvement=improvement,
        wmape_baseline_longtail=wmape_baseline_lt,
        wmape_model_longtail=wmape_model_lt,
        improvement_longtail=improvement_lt,
        n_sku_months=len(joined),
    )


def report_gate(result: WmapeResult) -> str:
    status = "PASS" if result.passes_gate() else "FAIL"
    return (
        f"demand wmape: model={result.wmape_model:.3f} baseline={result.wmape_baseline:.3f} "
        f"improvement={result.improvement:.3f} | long-tail "
        f"model={result.wmape_model_longtail:.3f} baseline={result.wmape_baseline_longtail:.3f} "
        f"improvement={result.improvement_longtail:.3f} ({status}, gate >= {WMAPE_GATE})"
    )
