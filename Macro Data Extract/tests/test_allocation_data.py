"""Focused checks for the allocation study's data-preparation helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "Investment Research"
    / "Statistical Learning in Asset Allocation"
)
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "allocation_data", DIRECTORY / "allocation_data.py"
)
assert spec is not None and spec.loader is not None
ad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad)


def test_to_daily_applies_lag_and_stops_dead_series():
    bdays = pd.bdate_range("2020-01-01", "2021-12-31")
    observations = pd.Series(
        [1.0, 2.0], index=pd.to_datetime(["2020-03-31", "2020-04-30"])
    )
    daily = ad.to_daily(observations, lag_days=14, bdays=bdays, freq="M")
    # Nothing is known before the first observation plus its publication lag.
    assert daily.loc["2020-04-13"] != daily.loc["2020-04-13"]  # NaN
    assert daily.loc["2020-04-14"] == 1.0
    assert daily.loc["2020-05-14"] == 2.0
    # A discontinued monthly series stops within its fill limit instead of
    # extending stale values indefinitely.
    assert np.isnan(daily.loc["2021-12-31"])


def test_growth_transforms_match_hand_calculations():
    index = pd.date_range("2019-01-31", periods=30, freq="ME")
    level = pd.Series(100.0 * 1.01 ** np.arange(30), index=index)
    growth = ad.yoy(level, 12)
    assert np.isclose(growth.dropna().iloc[0], 100.0 * (1.01**12 - 1.0))
    annualized = ad.ar_6m6m(level).dropna()
    # Constant 1% monthly growth annualizes to the same rate on 6m/6m windows.
    assert np.isclose(annualized.iloc[-1], 100.0 * (1.01**12 - 1.0), rtol=1e-6)
    change = ad.d3m3ml3(pd.Series(np.arange(30, dtype=float), index=index))
    assert np.isclose(change.dropna().iloc[0], 3.0)


def test_trend_and_median_windows():
    index = pd.date_range("2000-03-31", periods=40, freq="QE")
    quarterly = pd.Series(np.linspace(0.0, 3.9, 40), index=index)
    trend = ad.trend_20q(quarterly)
    assert np.isclose(trend.iloc[-1], quarterly.iloc[-20:].mean())
    monthly = pd.Series(
        np.arange(80, dtype=float),
        index=pd.date_range("2015-01-31", periods=80, freq="ME"),
    )
    gap = ad.vs_5y_median(monthly, 12)
    assert np.isclose(gap.iloc[-1], monthly.iloc[-1] - monthly.iloc[-60:].median())


def test_sleeve_map_is_complete_and_unique():
    assert len(ad.SLEEVES) == 11
    symbols = [symbol for symbol, _ in ad.SLEEVES.values()]
    assert len(set(symbols)) == 11
    assert ad.INFLATION_TARGETS["USD"] == 2.0


def test_series_map_never_interleaves_sources():
    dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"])
    raw = pd.DataFrame({
        "provider": "BIS", "cid": "AUD", "family": "CREDITGDP", "freq": "Q",
        "lag": 90, "obs_date": list(dates) * 2,
        "series": ["long"] * 3 + ["other"] * 3,
        "value": [150.0, 151.0, 152.0, 5000.0, 5001.0, 5002.0],
    })
    raw = pd.concat([raw, raw.iloc[[0]].assign(
        series="long", obs_date=pd.Timestamp("2019-12-31"), value=149.0)])
    series = ad.series_map(raw)[("AUD", "CREDITGDP")][0]
    # The longest source leads; the other never overwrites its dates.
    assert series.tolist() == [149.0, 150.0, 151.0, 152.0]


def test_bond_excess_matches_duration_approximation():
    index = pd.bdate_range("2020-01-01", periods=3)
    yld = pd.Series([2.0, 2.10, 2.10], index=index)
    cash = pd.Series(0.0, index=index)
    xr = ad.bond_excess_from_yield(yld, cash, tenor=5.0)
    # A 10bp rise on a 5-year par bond near 2% costs about 0.47% of price.
    assert -0.50 < xr.iloc[0] < -0.44
    # With no yield move and zero cash, the return is the yield accrual.
    days = (index[2] - index[1]).days
    assert np.isclose(xr.iloc[1], 2.10 * days / 365.0, rtol=1e-6)


def test_fx_forward_leg_adds_rate_gap():
    index = pd.bdate_range("2020-01-01", periods=5)
    spot = pd.DataFrame({code: 1.0 for code, _ in ad.FX_SPOT.values()}, index=index)
    policy = {cid: pd.Series(3.0, index=index) for cid in ad.FX_SPOT}
    dff = pd.Series(1.0, index=index)
    fx = ad.fx_forward_excess(spot, policy, dff)
    # Flat spot: each period earns the 2-point rate gap on an act/360 basis.
    days = index.to_series().diff().dt.days.iloc[1:]
    assert np.allclose(fx.loc[days.index].values, 2.0 * days.values / 360.0)
