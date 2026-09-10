"""未来情報の混入（ルックアヘッド）を検査する（VR-002）。

考え方: データを途中で打ち切っても、打ち切り点より前の判断は 1 つも変わらないはずである。
変わるなら、その戦略はどこかで未来を見ている。
"""
from pathlib import Path

import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.spec import load_all
from synthetic import flat_then_breakout_then_crash, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
SPECS = {s.id: s for s in load_all(CATALOG)}


def _entries(result):
    return [
        (t.symbol, t.entry_date, round(t.entry_price, 6), t.shares, round(t.stop_price, 6))
        for t in result.trades
    ]


@pytest.mark.parametrize("strategy_id", ["ST-06", "ST-12"])
def test_truncating_data_does_not_change_past_decisions(strategy_id):
    spec = SPECS[strategy_id]
    df = flat_then_breakout_then_crash(n_flat=250, n_up=40, n_down=60)
    idx = rising_index(len(df), start=str(df.index[0].date()))
    cfg = BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0)

    full = run(spec, {"TEST": df}, index=idx, config=cfg)
    cut = df.index[int(len(df) * 0.7)]
    part = run(spec, {"TEST": df.loc[:cut]}, index=idx, config=cfg)

    # 打ち切り点より前に建てた玉は、両者で完全に一致しなければならない。
    # （最終日の強制手仕舞いは打ち切りで変わるため、エントリーだけを比べる）
    full_before = [e for e in _entries(full) if e[1] < cut]
    part_before = [e for e in _entries(part) if e[1] < cut]
    assert full_before == part_before, (
        f"{strategy_id}: 打ち切りで過去の判断が変わった。未来を参照している疑いがある"
    )


@pytest.mark.parametrize("strategy_id", ["ST-06", "ST-12"])
def test_future_bars_cannot_affect_signals(strategy_id):
    """打ち切り点より後のデータを改変しても、それ以前のエントリーは変わらない。"""
    spec = SPECS[strategy_id]
    df = flat_then_breakout_then_crash(n_flat=250, n_up=40, n_down=60)
    idx = rising_index(len(df), start=str(df.index[0].date()))
    cfg = BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0)

    cut_pos = int(len(df) * 0.7)
    cut = df.index[cut_pos]
    tampered = df.copy()
    tampered.iloc[cut_pos:] = tampered.iloc[cut_pos:] * 3.0   # 未来だけを大きく歪める

    base = [e for e in _entries(run(spec, {"TEST": df}, index=idx, config=cfg)) if e[1] <= cut]
    after = [e for e in _entries(run(spec, {"TEST": tampered}, index=idx, config=cfg)) if e[1] <= cut]
    assert base == after, f"{strategy_id}: 未来のバーが過去のシグナルに影響している"


def test_entry_never_uses_the_signal_day_close_as_fill():
    """約定価格が判定日の終値と一致してはならない（翌営業日の始値で約定する）。"""
    spec = SPECS["ST-06"]
    df = flat_then_breakout_then_crash(n_flat=250, n_up=40, n_down=60)
    idx = rising_index(len(df), start=str(df.index[0].date()))
    res = run(spec, {"TEST": df}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    assert res.trades
    for t in res.trades:
        pos = df.index.get_loc(t.entry_date)
        assert pos > 0
        assert t.entry_price == pytest.approx(df.at[t.entry_date, "open"])
        assert t.entry_price != pytest.approx(df.iloc[pos - 1]["close"])
