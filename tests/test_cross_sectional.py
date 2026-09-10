"""クロスセクショナル順位（銘柄間の相対順位）の検査。ST-09 / ST-04 が依存する。"""
from pathlib import Path

import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.spec import load_all
from synthetic import make_bars, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
SPECS = {s.id: s for s in load_all(CATALOG)}


def _universe():
    """強い銘柄・普通・弱い銘柄の3本。ST-09 は STRONG だけを買うはず。"""
    n = 400
    strong = make_bars([100 * (1.004 ** i) for i in range(n)])
    mid = make_bars([100 * (1.001 ** i) for i in range(n)])
    weak = make_bars([100 * (0.999 ** i) for i in range(n)])
    return {"STRONG": strong, "MID": mid, "WEAK": weak}


def test_momentum_buys_only_the_strongest():
    data = _universe()
    idx = rising_index(400, start=str(data["STRONG"].index[0].date()))
    res = run(SPECS["ST-09"], data, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    assert res.trades, f"エントリーなし: {res.rejections}"
    assert {t.symbol for t in res.trades} == {"STRONG"}


def test_rank_is_relative_not_absolute():
    """全銘柄が同じだけ上がると、順位では差がつかない＝誰も上位10%にならない。"""
    n = 400
    same = {c: make_bars([100 * (1.004 ** i) for i in range(n)]) for c in ("A", "B", "C")}
    idx = rising_index(n, start=str(same["A"].index[0].date()))
    res = run(SPECS["ST-09"], same, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    # 同値のランクは平均順位になるため、上位10%（>=90）には誰も入らない
    assert not res.trades
