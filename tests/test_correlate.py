"""相関測定（VR-045）の検査。"""
from pathlib import Path

import numpy as np
import pandas as pd

from short_trade.correlate import format_table, measure
from short_trade.spec import load_all
from synthetic import make_bars, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
SPECS = {s.id: s for s in load_all(CATALOG)}


def _universe(n=420):
    rng = np.random.default_rng(7)
    data = {}
    for i, code in enumerate(("A", "B", "C", "D", "E")):
        drift = 1.0015 + i * 0.0004
        noise = 1 + rng.normal(0, 0.012, n)
        closes = 1000 * np.cumprod(np.full(n, drift) * noise)
        data[code] = make_bars(closes)
    return data


def test_measure_reports_pairs_predictions_and_metrics():
    data = _universe()
    idx = rising_index(420, start=str(data["A"].index[0].date()))
    rep = measure([SPECS["ST-06"], SPECS["ST-12"], SPECS["ST-09"]], data, index=idx)
    assert set(rep.strategies) <= {"ST-06", "ST-12", "ST-09"}
    assert len(rep.strategies) >= 2
    d = rep.to_dict()
    assert d["pairs"] and all("return_corr" in p for p in d["pairs"])
    assert rep.predicted.get(("ST-06", "ST-12")) == 0.0          # docs/18 §18.4 の事前予想
    assert all(sid in rep.metrics for sid in rep.strategies)
    assert "ペア" in format_table(rep)


def test_identical_strategy_correlates_perfectly_with_itself():
    """同じ仕様を2本渡せば相関は 1.0。測定器の健全性確認。"""
    import copy
    data = _universe()
    idx = rising_index(420, start=str(data["A"].index[0].date()))
    twin = copy.deepcopy(SPECS["ST-06"])
    twin.id = "ST-06b"
    rep = measure([SPECS["ST-06"], twin], data, index=idx)
    assert rep.return_corr.loc["ST-06", "ST-06b"] > 0.999
    assert rep.entry_jaccard.loc["ST-06", "ST-06b"] == 1.0
    assert rep.flags() and rep.flags()[0][2] > 0.999


def test_unsupported_strategy_is_skipped_not_fatal():
    data = _universe()
    idx = rising_index(420, start=str(data["A"].index[0].date()))
    rep = measure([SPECS["ST-06"], SPECS["ST-23"]], data, index=idx)
    assert "ST-23" in rep.skipped and "ST-06" in rep.strategies


def test_residual_correlation_removes_market_beta():
    """指数と同じ動きをする2本は、生の相関は高いが残差相関はそれより低い。

    ここでは測定器の性質だけを確かめる: 残差行列が出ること、同一戦略の残差相関は1のままであること、
    ベータが各戦略ぶん計算されること。
    """
    import copy
    data = _universe()
    idx = rising_index(420, start=str(data["A"].index[0].date()))
    twin = copy.deepcopy(SPECS["ST-06"])
    twin.id = "ST-06b"
    rep = measure([SPECS["ST-06"], twin], data, index=idx)
    assert rep.residual_corr is not None
    assert rep.residual_corr.loc["ST-06", "ST-06b"] > 0.999
    assert set(rep.betas) == {"ST-06", "ST-06b"}
    d = rep.to_dict()
    assert d["judged_on"] == "residual"
    assert d["pairs"][0]["residual_corr"] is not None
    assert d["flags_over_0.7"] and d["raw_flags_over_0.7"]


def test_market_residuals_zero_out_a_pure_beta_curve():
    """戦略リターン = 2×指数リターン なら、残差はゼロでベータは2。"""
    from short_trade.correlate import market_residuals
    idx = rising_index(50, start="2024-01-01")
    m = idx["close"].pct_change().fillna(0.0)
    curves = {"X": 2.0 * m}
    res, betas = market_residuals(curves, idx)
    assert abs(betas["X"] - 2.0) < 1e-9
    assert float(res["X"].abs().max()) < 1e-12
