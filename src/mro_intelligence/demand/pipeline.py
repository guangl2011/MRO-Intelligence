"""Long-tail demand sensing pipeline: forecasts, health cards, WMAPE metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mro_intelligence.demand import models
from mro_intelligence.demand import series as srs
from mro_intelligence.demand.evaluate import WmapeResult, evaluate_wmape

QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


def category_seasonal_index(train: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Pooled month-of-year demand factors per category, normalized to mean 1.

    A SKU-level index sees only ~2 samples per calendar month and is mostly
    noise; pooling across a category's hundreds of SKUs recovers the shared
    HVAC/seasonal calendar the ERP flat-rate baseline cannot see. Missing
    (category, month) pairs fall back to 1.0.
    """
    if "category" not in train.columns:
        return {}
    df = train.copy()
    df["cal"] = df["month"].str.slice(5, 7).astype(int)
    month_mean = df.groupby(["category", "cal"])["qty"].mean()
    cat_mean = df.groupby("category")["qty"].mean()
    index = (month_mean / cat_mean).fillna(1.0)
    return {(str(cat), int(cal)): float(v) for (cat, cal), v in index.items()}


def forecast_horizon(
    train: pd.DataFrame,
    horizon_months: list[str],
    categories_by_sku: dict[str, str],
    index: dict[tuple[str, int], float],
    series: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-(sku, month) forecasts: model rate x category seasonal factor."""
    forecasts = build_forecasts(train, series)
    rates = dict(zip(forecasts["sku"].astype(str), forecasts["point_forecast"], strict=True))
    rows = []
    for month in horizon_months:
        cal = int(month[5:7])
        for sku, rate in rates.items():
            category = categories_by_sku.get(sku)
            factor = index.get((str(category), cal), 1.0) if category is not None else 1.0
            rows.append({"sku": sku, "month": month, "forecast": rate * factor})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class DemandOutputs:
    """One tenant's demand-sensing results."""

    tenant_id: str
    forecasts: pd.DataFrame
    health_report: pd.DataFrame
    metrics: WmapeResult | None


def _nonzero_history(values: np.ndarray, lookback: int = 24) -> np.ndarray:
    """Recent nonzero demands, newest last; empty when the SKU never sold."""
    recent = values[-lookback:]
    return recent[recent > 0]


def _forecast_row(sku: str, values: np.ndarray) -> dict[str, float | str]:
    """Fit the selected model and compose the distribution for one SKU."""
    choice = models.choose_model(values)
    probability, size = models.tsb(values) if choice == "tsb" else models.sba(values)
    band = models.quantile_band(probability, size, _nonzero_history(values), QUANTILE_LEVELS)
    row: dict[str, float | str] = {
        "sku": sku,
        "model": choice,
        "sbc_class": models.demand_class(values),
        "occurrence_probability": round(probability, 6),
        "size_mean": round(size, 4),
        "point_forecast": round(probability * size, 6),
    }
    for tau in QUANTILE_LEVELS:
        row[f"q{int(tau * 100):02d}"] = round(band[tau], 4)
    return row


def build_forecasts(
    train: pd.DataFrame, series: dict[str, np.ndarray] | None = None
) -> pd.DataFrame:
    """Per-SKU next-month forecast from train-month demand vectors."""
    vectors = srs.series_by_sku(train) if series is None else series
    rows = [_forecast_row(sku, values) for sku, values in sorted(vectors.items())]
    return pd.DataFrame(rows)


def _trailing_demand(train: pd.DataFrame, months: int) -> pd.DataFrame:
    """Total demand per SKU over the last `months` train months."""
    trailing = sorted(train["month"].unique())[-months:]
    recent = train[train["month"].isin(trailing)]
    return recent.groupby("sku", as_index=False)["qty"].sum().rename(columns={"qty": "demand_12m"})


def build_health_report(
    train: pd.DataFrame,
    forecasts: pd.DataFrame,
    segmentation: pd.DataFrame,
    on_hand: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecasts, classes, and stock position into the SDD health card."""
    report = forecasts.merge(segmentation[["sku", "abc_class", "xyz_class"]], on="sku", how="left")
    report = report.merge(on_hand, on="sku", how="left").merge(
        _trailing_demand(train, srs.DEAD_STOCK_MONTHS), on="sku", how="left"
    )
    report["on_hand"] = report["on_hand"].fillna(0.0)
    report["demand_12m"] = report["demand_12m"].fillna(0.0)
    report["months_of_cover"] = [
        srs.cover_months(float(o), float(f))
        for o, f in zip(report["on_hand"], report["point_forecast"], strict=True)
    ]
    report["dead_stock"] = [
        srs.dead_stock_flag(float(d), float(o))
        for d, o in zip(report["demand_12m"], report["on_hand"], strict=True)
    ]
    report["stockout_risk"] = [
        srs.stockout_risk(float(o), float(q25), float(q75), float(q95))
        for o, q25, q75, q95 in zip(
            report["on_hand"],
            report["q25"],
            report["q75"],
            report["q95"],
            strict=True,
        )
    ]
    return report


def run_demand(
    extracts: dict[str, pd.DataFrame],
    segmentation: pd.DataFrame,
    tenant_id: str,
    *,
    evaluate: bool = False,
) -> DemandOutputs:
    """Forecast next-month demand per SKU and assemble the SDD health card.

    The latest months are held out when evaluating: models train on the
    remaining history, then WMAPE compares the seasonal intermittent forecast
    against the ERP-style flat rate on the same held-out months.
    """
    demand = srs.monthly_demand(extracts["order_lines"])
    if "item_master" in extracts:
        demand = demand.merge(extracts["item_master"][["sku", "category"]], on="sku", how="left")
    train, eval_df = srs.horizon_split(demand)

    # One pass over the train frame feeds the forecaster, the horizon
    # evaluator, and the baseline rows; recomputing per-SKU series per call
    # is quadratic in tenant size.
    series = srs.series_by_sku(train)
    forecasts = build_forecasts(train, series)
    on_hand = srs.latest_on_hand(extracts["inventory_monthly"])
    health = build_health_report(train, forecasts, segmentation, on_hand)

    metrics: WmapeResult | None = None
    if evaluate:
        categories_by_sku = (
            dict(zip(train["sku"].astype(str), train["category"].astype(str), strict=True))
            if "category" in train.columns
            else {}
        )
        index = category_seasonal_index(train)
        model = forecast_horizon(
            train, sorted(eval_df["month"].unique()), categories_by_sku, index, series
        )
        horizon_months = sorted(eval_df["month"].unique())
        # The ERP baseline rate is a flat trailing mean: month-invariant per
        # SKU, so it is computed once and repeated across horizon months.
        baseline_rate = {sku: models.moving_average(values) for sku, values in series.items()}
        baseline_rows = [
            {
                "sku": sku,
                "month": month,
                "forecast": baseline_rate[sku],
            }
            for sku in sorted(baseline_rate)
            for month in horizon_months
        ]
        metrics = evaluate_wmape(eval_df, model, pd.DataFrame(baseline_rows), segmentation)
    return DemandOutputs(
        tenant_id=tenant_id, forecasts=forecasts, health_report=health, metrics=metrics
    )


def write_demand(outputs: DemandOutputs, out_dir: Path) -> dict[str, Path]:
    """Write forecasts, the health report, and metrics under out_dir/tenant."""
    tenant_dir = out_dir / outputs.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": tenant_dir / "demand_forecasts.csv",
        "health": tenant_dir / "sku_health_report.csv",
    }
    outputs.forecasts.to_csv(paths["forecasts"], index=False)
    outputs.health_report.to_csv(paths["health"], index=False)
    if outputs.metrics is not None:
        metric_path = tenant_dir / "demand_metrics.csv"
        pd.DataFrame(
            [
                {
                    "wmape_model": outputs.metrics.wmape_model,
                    "wmape_baseline": outputs.metrics.wmape_baseline,
                    "improvement": outputs.metrics.improvement,
                    "wmape_model_longtail": outputs.metrics.wmape_model_longtail,
                    "wmape_baseline_longtail": outputs.metrics.wmape_baseline_longtail,
                    "improvement_longtail": outputs.metrics.improvement_longtail,
                    "n_sku_months": outputs.metrics.n_sku_months,
                    "gate_pass": outputs.metrics.passes_gate(),
                }
            ]
        ).to_csv(metric_path, index=False)
        paths["metrics"] = metric_path
    return paths
