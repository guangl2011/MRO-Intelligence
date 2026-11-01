"""Unit tests for the M3 demand-sensing module."""

from __future__ import annotations

import pandas as pd
import pytest

from mro_intelligence.demand import models
from mro_intelligence.demand.evaluate import evaluate_wmape
from mro_intelligence.demand.pipeline import (
    build_forecasts,
    category_seasonal_index,
    forecast_horizon,
    run_demand,
)
from mro_intelligence.demand.series import (
    cover_months,
    dead_stock_flag,
    horizon_split,
    latest_on_hand,
    monthly_demand,
    series_by_sku,
    series_for,
    stockout_risk,
)


def test_series_by_sku_matches_per_sku_series() -> None:
    # One row per (sku, month): the groupby pass must return exactly what
    # the per-SKU scan returns, for every SKU.
    train = pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b", "c"],
            "month": ["2026-01", "2026-02", "2026-01", "2026-02", "2026-01"],
            "qty": [1.0, 4.0, 2.0, 0.0, 7.0],
        }
    )
    vectors = series_by_sku(train)
    assert sorted(vectors) == ["a", "b", "c"]
    for sku in ("a", "b", "c"):
        assert (vectors[sku] == series_for(train, sku)).all()


def test_croston_tracks_intermittent_series() -> None:
    # Occurrences at months 1 and 4 (5 and 3 units). Bounded alpha: the first
    # update uses 0.3 (clip(1/1)), so size = 5 + 0.3*(3-5) and the interval
    # smooths 2 -> 2 + 0.3*(3-2).
    probability, size = models.croston([0, 5, 0, 0, 3, 0, 0])
    assert size == pytest.approx(4.4)
    assert probability == pytest.approx(1 / 2.3)
    assert probability * size == pytest.approx(4.4 / 2.3)


def test_sba_bias_corrects_below_croston() -> None:
    series = [0, 5, 0, 0, 3, 0, 0]
    p_croston, size_croston = models.croston(series)
    p_sba, _ = models.sba(series)
    assert p_sba < p_croston
    # SBA scales the per-period rate, not the size estimate.
    _, size_sba = models.sba(series)
    assert size_sba == size_croston


def test_tsb_probability_updates_every_period() -> None:
    # Zero tail pulls the occurrence probability down even without demand.
    p_after_demand, _ = models.tsb([0, 0, 4, 0])
    p_with_tail, _ = models.tsb([0, 0, 4, 0, 0, 0, 0, 0])
    assert p_with_tail < p_after_demand


def test_moving_average_is_the_flat_erp_rate() -> None:
    assert models.moving_average([1, 2, 3, 4, 5, 6], window=3) == pytest.approx(5.0)


def test_demand_class_quadrants() -> None:
    assert models.demand_class([4, 5, 4, 5, 4, 5, 4, 5]) == "smooth"
    # Rare but consistent sizes: intermittent, not lumpy.
    assert models.demand_class([0, 0, 0, 0, 2, 0, 0, 0]) == "intermittent"
    # Rare AND wildly varying sizes: lumpy.
    assert models.demand_class([0, 0, 0, 0, 2, 0, 0, 20]) == "lumpy"
    assert models.demand_class([0, 0, 0, 0, 0, 0, 0, 0]) == "lumpy"


def test_choose_model_prefers_tsb_on_zero_heavy_series() -> None:
    assert models.choose_model([0, 0, 0, 0, 2, 0, 0, 0]) == "tsb"
    assert models.choose_model([4, 5, 4, 5, 4, 5, 4, 5]) == "sba"


def test_quantile_band_zero_mass_and_monotonicity() -> None:
    band = models.quantile_band(probability=0.4, size=5.0, nonzero_history=[2, 4, 6, 8, 10])
    # 60% zero mass: the 5th and 25th percentiles sit inside it.
    assert band[0.05] == 0.0
    assert band[0.25] == 0.0
    values = [band[t] for t in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert values == sorted(values)
    assert band[0.95] <= 10  # never exceeds the size history


def test_monthly_demand_zero_fills_missing_months() -> None:
    lines = pd.DataFrame(
        {
            "order_date": ["2024-01-05", "2024-03-02"],
            "sku": ["S1", "S1"],
            "qty": [2.0, 1.0],
        }
    )
    demand = monthly_demand(lines)
    assert list(demand["month"]) == ["2024-01", "2024-02", "2024-03"]
    assert list(demand["qty"]) == [2.0, 0.0, 1.0]


def test_horizon_split_holds_out_latest_months() -> None:
    demand = pd.DataFrame(
        {
            "sku": ["S1"] * 5,
            "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
            "qty": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    train, eval_df = horizon_split(demand, horizon_months=2)
    assert set(train["month"]) == {"2024-01", "2024-02", "2024-03"}
    assert set(eval_df["month"]) == {"2024-04", "2024-05"}
    with pytest.raises(ValueError, match="months of demand history"):
        horizon_split(demand.head(2))


def test_latest_on_hand_sums_branches_of_last_snapshot() -> None:
    snap = pd.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "sku": ["S1", "S1", "S1", "S2"],
            "on_hand": [3.0, 4.0, 5.0, 0.0],
        }
    )
    on_hand = latest_on_hand(snap).set_index("sku")["on_hand"]
    assert on_hand["S1"] == 5.0  # latest month only, both branches summed
    assert on_hand["S2"] == 0.0


def test_health_flags_follow_sdd_definitions() -> None:
    assert dead_stock_flag(last12=0.0, on_hand=3.0)
    assert not dead_stock_flag(last12=2.0, on_hand=3.0)
    assert not dead_stock_flag(last12=0.0, on_hand=0.0)
    assert cover_months(on_hand=10.0, monthly_forecast=4.0) == pytest.approx(2.5)
    assert cover_months(on_hand=10.0, monthly_forecast=0.0) == float("inf")
    assert stockout_risk(2.0, 5.0, 10.0, 20.0) == "high"
    assert stockout_risk(7.0, 5.0, 10.0, 20.0) == "medium"
    assert stockout_risk(15.0, 5.0, 10.0, 20.0) == "low"
    assert stockout_risk(30.0, 5.0, 10.0, 20.0) == "none"


def _seg_frame(skus: list[str], xyz: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": skus,
            "abc_class": ["A"] * len(skus),
            "xyz_class": xyz,
        }
    )


def test_evaluate_wmape_scores_improvement_against_baseline() -> None:
    eval_df = pd.DataFrame(
        {"sku": ["S1", "S2"], "month": ["2024-04", "2024-04"], "qty": [10.0, 1.0]}
    )
    model = pd.DataFrame(
        {"sku": ["S1", "S2"], "month": ["2024-04", "2024-04"], "forecast": [10.0, 0.0]}
    )
    # Baseline flat-rates 5 onto the low-demand tail SKU.
    baseline = pd.DataFrame(
        {"sku": ["S1", "S2"], "month": ["2024-04", "2024-04"], "forecast": [10.0, 5.0]}
    )
    result = evaluate_wmape(eval_df, model, baseline, _seg_frame(["S1", "S2"], ["X", "Z"]))
    assert result.wmape_model == pytest.approx(1.0 / 11.0)
    assert result.wmape_baseline == pytest.approx(4.0 / 11.0)  # |5-1| error on S2
    assert result.improvement == pytest.approx(0.75)
    # The Z-quadrant carries the same win, so the gate passes.
    assert result.passes_gate()


def test_evaluate_wmape_gate_scores_on_long_tail_only() -> None:
    # The model wins on one X-class row but loses on the Z-quadrant rows that
    # carry the S2 gate: the gate verdict must follow the Z rows.
    eval_df = pd.DataFrame(
        {
            "sku": ["FAST", "SLOW", "SLOW"],
            "month": ["2024-04", "2024-04", "2024-05"],
            "qty": [100.0, 0.0, 2.0],
        }
    )
    model = pd.DataFrame(
        {
            "sku": ["FAST", "SLOW", "SLOW"],
            "month": ["2024-04", "2024-04", "2024-05"],
            "forecast": [50.0, 2.5, 2.5],
        }
    )
    baseline = pd.DataFrame(
        {
            "sku": ["FAST", "SLOW", "SLOW"],
            "month": ["2024-04", "2024-04", "2024-05"],
            "forecast": [200.0, 1.0, 1.0],
        }
    )
    result = evaluate_wmape(eval_df, model, baseline, _seg_frame(["FAST", "SLOW"], ["X", "Z"]))
    # The model wins overall (half the FAST error, tail rows comparable) but
    # the zero-heavy tail rows that carry the S2 gate favor the baseline.
    assert result.improvement > 0
    assert result.improvement_longtail < 0
    assert not result.passes_gate()


def test_category_seasonal_index_recovers_shared_pattern() -> None:
    # COOL peaks in July, FLAT never varies; two years of history.
    months = [f"202{y}-{m:02d}" for y in (3, 4) for m in range(1, 13)]
    demand = pd.DataFrame(
        {
            "sku": ["C1", "F1"] * len(months),
            "month": months * 2,
            "qty": [(10.0 * 2 if m == 7 else 10.0) for m in range(1, 13)] * 2
            + [10.0] * len(months),
            "category": ["COOL"] * len(months) + ["FLAT"] * len(months),
        }
    )
    index = category_seasonal_index(demand)
    # Factor = July mean / category-wide mean (July itself included).
    cool_all_mean = (20.0 * 2 + 10.0 * 22) / 24
    assert index[("COOL", 7)] == pytest.approx(20.0 / cool_all_mean)
    assert index[("FLAT", 7)] == pytest.approx(1.0)
    # Unknown pairs fall back to neutral in the horizon forecast.
    assert index.get(("GHOST", 7), 1.0) == 1.0


def test_forecast_horizon_scales_rate_by_category_factor() -> None:
    months = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 13)]
    train = pd.DataFrame(
        {
            "sku": ["C1", "C2"] * len(months),
            "month": months * 2,
            "qty": [10.0] * len(months) * 2,
            "category": ["COOL"] * len(months) * 2,
        }
    )
    # C2 sells double C1's rate; both share the category calendar.
    train.loc[train["sku"] == "C2", "qty"] = 20.0
    index = {("COOL", 7): 2.0, ("COOL", 8): 1.0}
    horizon = forecast_horizon(
        train, ["2025-07", "2025-08"], {"C1": "COOL", "C2": "COOL"}, index
    ).set_index(["sku", "month"])
    # Constant series: SBA rate = (1 - alpha/2) * mean = 0.975 * level,
    # then scaled by the category calendar factor.
    assert horizon.loc[("C1", "2025-07"), "forecast"] == pytest.approx(19.5)
    assert horizon.loc[("C2", "2025-07"), "forecast"] == pytest.approx(39.0)
    assert horizon.loc[("C1", "2025-08"), "forecast"] == pytest.approx(9.75)


def test_run_demand_end_to_end_minimal() -> None:
    # 15 months, two SKUs: a steady A-mover and an all-zero tail SKU.
    months = pd.period_range("2023-01", "2024-03", freq="M")
    lines = pd.DataFrame(
        {
            "order_date": [str(m) + "-01" for m in months for _ in range(2)],
            "sku": ["S1", "S2"] * len(months),
            "qty": [4.0, 0.0] * len(months),
        }
    )
    extracts = {
        "order_lines": lines,
        "inventory_monthly": pd.DataFrame(
            {
                "month": ["2024-03-01", "2024-03-01"],
                "sku": ["S1", "S2"],
                "on_hand": [9.0, 7.0],
            }
        ),
    }
    segmentation = pd.DataFrame(
        {"sku": ["S1", "S2"], "abc_class": ["A", "C"], "xyz_class": ["X", "Z"]}
    )
    outputs = run_demand(extracts, segmentation, tenant_id="t1", evaluate=True)
    f = outputs.forecasts.set_index("sku")
    assert f.loc["S1", "point_forecast"] == pytest.approx(4.0, abs=0.3)
    assert f.loc["S2", "point_forecast"] == pytest.approx(0.0, abs=0.2)
    health = outputs.health_report.set_index("sku")
    # Stocked with zero demand: dead stock per the SDD definition.
    assert bool(health.loc["S2", "dead_stock"]) is True
    assert health.loc["S1", "stockout_risk"] == "none"  # 9 on hand vs ~4/month
    assert outputs.metrics is not None
    assert outputs.metrics.n_sku_months == 6  # 2 SKUs x 3 held-out months


def test_build_forecasts_deterministic_for_identical_series() -> None:
    train = pd.DataFrame(
        {
            "sku": ["A", "B"] * 6,
            "month": ["2024-01", "2024-02", "2024-03"] * 4,
            "qty": [2.0, 1.0] * 6,
        }
    )
    first = build_forecasts(train)
    second = build_forecasts(train)
    pd.testing.assert_frame_equal(first, second)
