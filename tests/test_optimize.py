"""学習／検証の分割つき探索（optimize）とパラメータ置換（spec.with_params）の検査。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.optimize import g1_check, grid_levels, walk_forward
from short_trade.spec import SpecError, load_all
from synthetic import flat_then_breakout, make_bars, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
SPECS = {s.id: s for s in load_all(CATALOG)}


def test_grid_levels_are_range_ends_and_default_only():
    assert grid_levels({"name": "x", "default": 3.0, "range": [2.0, 5.0], "step": 0.5}) == [2.0, 3.0, 5.0]
    assert grid_levels({"name": "n", "default": 2, "range": [2, 4], "step": 1}) == [2, 3, 4]   # 既定が端なら中点
    assert grid_levels({"name": "k", "default": 21, "range": [0, 42], "step": 21}) == [0, 21, 42]


def test_with_params_rejects_undeclared_parameter():
    with pytest.raises(SpecError):
        SPECS["ST-06"].with_params(atr_mult=3)


def test_with_params_keeps_template_and_substitutes_arithmetic():
    st09 = SPECS["ST-09"].with_params(lookback_days=63, skip_days=0, top_pct=25)
    assert st09.raw["definitions"]["momentum_score"] == "close[-0] / close[-63] - 1"
    assert st09.entry_conditions[0].endswith(">= 75")
    again = st09.with_params(top_pct=5)                    # 置換済みの仕様からでも原本に戻して再置換
    assert again.entry_conditions[0].endswith(">= 95") and again.params["lookback_days"] == 63
    assert "{{" not in str(again.raw["exit"])


def test_defaults_reproduce_original_expressions():
    assert SPECS["ST-06"].entry_conditions[0] == "close > HIGHEST(high, 20)[-1]"
    assert SPECS["ST-12"].entry_conditions[1] == "RSI(2) < 10"
    assert SPECS["ST-04"].stop_initial == "max(vcp.p3_low, close_at_entry * (1 - 3 / 100))"


def test_entries_from_blocks_entries_before_the_date():
    spec = SPECS["ST-06"]
    df = flat_then_breakout()
    idx = rising_index(len(df), start=str(df.index[0].date()))
    base = dict(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1, max_position_pct=100)
    normal = run(spec, {"T": df}, index=idx, config=BacktestConfig(**base))
    blocked = run(spec, {"T": df}, index=idx, config=BacktestConfig(**base, entries_from=df.index[-1]))
    assert normal.trades and not blocked.trades


def _universe(n=900, seed=5):
    rng = np.random.default_rng(seed)
    data = {}
    for i, code in enumerate("ABCDEF"):
        closes = 1000 * np.cumprod((1.0006 + i * 0.0002) * (1 + rng.normal(0, 0.012, n)))
        data[code] = make_bars(closes, start="2020-01-01")
    return data


def test_walk_forward_reports_grid_best_and_test_for_both(monkeypatch):
    data = _universe()
    idx = rising_index(900, start="2020-01-01")
    spec = SPECS["ST-06"]
    r = walk_forward(spec, data, index=idx, train_end="2022-06-30", equity=1_000_000, slippage_pct=0.0,
                     min_trades=5)
    assert len(r.grid) == 9                                   # 2 パラメータ × 3 水準
    assert set(r.best) == {"entry_lookback", "exit_lookback"}
    assert r.defaults == {"entry_lookback": 20, "exit_lookback": 10}
    assert set(r.test) == {"default", "best"}
    for v in r.test.values():
        assert "operating" in v and "structural" in v
    assert 0.0 <= r.plateau_share <= 1.0
    assert set(r.g1["default"]) == {"取引数 >= 100", "PF >= 1.3", "期待値 > 0", "最大DD <= 15%"}
    d = r.to_dict()
    assert d["train_end"] == "2022-06-30" and len(d["grid"]) == 9


def test_g1_check_thresholds():
    assert all(g1_check({"取引数": 100, "プロフィットファクタ": 1.3, "期待値": 0.1, "最大DD": -15.0}).values())
    assert not g1_check({"取引数": 99, "プロフィットファクタ": 2.0, "期待値": 1.0, "最大DD": -5.0})["取引数 >= 100"]
