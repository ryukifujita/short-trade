"""ST-04 の VCP 検出器の検査。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.detectors import vcp
from short_trade.spec import load_all
from synthetic import rising_index, vcp_pattern

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
ST04 = next(s for s in load_all(CATALOG) if s.id == "ST-04")


def test_detector_sees_three_shrinking_contractions_on_breakout_day():
    bars, bpos = vcp_pattern()
    out = vcp(bars)
    row = out.iloc[bpos]
    assert row["contractions"] >= 3
    assert row["p1"] > row["p2"] > row["p3"], (row["p1"], row["p2"], row["p3"])
    assert row["p3"] <= row["p1"] * 0.5
    assert row["v1"] > row["v2"] > row["v3"]
    assert 2.0 < row["p3"] < 4.5            # 最後の押しは約3%
    assert bars["close"].iloc[bpos] > row["pivot"]


def test_detector_is_causal():
    """データを打ち切っても、打ち切り点より前の検出結果は変わらない。"""
    bars, bpos = vcp_pattern()
    full = vcp(bars)
    cut = bpos - 5
    part = vcp(bars.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_dtype=False)


def test_pivot_is_not_visible_before_confirmation():
    """山は「反転が閾値を超えた日」に初めて見える。山の当日には見えてはいけない。"""
    bars, bpos = vcp_pattern()
    out = vcp(bars)
    peak3_pos = int(bars["close"].iloc[:bpos].idxmax() == bars.index[bpos - 1]) or None
    # 直近の山（peak3）の位置を探す
    p3 = bars["close"].iloc[bpos - 20:bpos].idxmax()
    pos = bars.index.get_loc(p3)
    assert out["pivot"].iloc[pos] != pytest.approx(bars["high"].iloc[pos]), "山の当日にその山が見えている"


def test_st04_enters_the_day_after_breakout():
    bars, bpos = vcp_pattern()
    idx = rising_index(len(bars), start=str(bars.index[0].date()))
    res = run(ST04, {"T": bars}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0, hysteresis_days=1))
    assert res.trades, f"ST-04 がエントリーしていない: {res.rejections}"
    t = res.trades[0]
    assert t.entry_date == bars.index[bpos + 1]
    assert t.entry_price == pytest.approx(bars["open"].iloc[bpos + 1])
    # 損切りは直近安値、ただし建値の3%以内（仕様の max(...)）
    assert t.stop_price >= t.entry_price * 0.97 - 1e-6
    assert t.stop_price < t.entry_price


def test_st04_does_not_enter_without_volume_confirmation():
    bars, bpos = vcp_pattern()
    # 直前20日の平均出来高は約0.45M（収縮で枯れている）。1.5倍未満の 0.5M にする
    bars.loc[bars.index[bpos], "volume"] = 500_000.0
    idx = rising_index(len(bars), start=str(bars.index[0].date()))
    res = run(ST04, {"T": bars}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0, hysteresis_days=1))
    assert not [t for t in res.trades if t.entry_date == bars.index[bpos + 1]]
