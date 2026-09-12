from pathlib import Path

import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.spec import load_all
from synthetic import flat_then_breakout, flat_then_breakout_then_crash, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"


@pytest.fixture(scope="module")
def st06():
    return next(s for s in load_all(CATALOG) if s.id == "ST-06")


def _index_for(df):
    return rising_index(len(df), start=str(df.index[0].date()))


def test_entry_fills_at_next_open_not_signal_close(st06):
    """判定は当日終値、約定は翌営業日の始値。ここがずれると全部が狂う。"""
    df = flat_then_breakout()
    res = run(st06, {"TEST": df}, index=_index_for(df),
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))

    assert res.trades, f"エントリーが発生していない: {res.rejections}"
    t = res.trades[0]
    signal_date = df.index[df.index.get_loc(t.entry_date) - 1]
    assert t.entry_price == pytest.approx(df.at[t.entry_date, "open"])
    assert t.entry_price != df.at[signal_date, "close"]


def test_stop_is_below_entry_and_from_ten_day_low(st06):
    df = flat_then_breakout()
    res = run(st06, {"TEST": df}, index=_index_for(df),
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    t = res.trades[0]
    signal_date = df.index[df.index.get_loc(t.entry_date) - 1]
    expected = df["low"].rolling(10).min().at[signal_date]
    assert t.stop_price == pytest.approx(expected)
    assert t.stop_price < t.entry_price


def test_stop_exit_happens_at_next_open(st06):
    """論理ストップ: 終値が割れた翌営業日の始値で成行決済される（RM-001a）。"""
    df = flat_then_breakout_then_crash()
    res = run(st06, {"TEST": df}, index=_index_for(df),
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    closed = [t for t in res.trades if t.exit_reason]
    assert closed
    t = closed[0]
    assert t.exit_reason in ("論理ストップ", "トレーリングストップ割れ")
    assert t.exit_price == pytest.approx(df.at[t.exit_date, "open"])


def test_position_size_respects_risk_and_position_cap(st06):
    df = flat_then_breakout()
    cfg = BacktestConfig(initial_equity=50_000, slippage_pct=0.0,
                         max_position_pct=15.0, assumed_loss_multiple=2.0)
    res = run(st06, {"TEST": df}, index=_index_for(df), config=cfg)
    t = res.trades[0]

    # 1銘柄あたりの投下額は資金の15%以内（RM-001c）
    assert t.entry_price * t.shares <= 50_000 * 0.15 * 1.02
    # 想定損失（損切り幅×2）は資金の1%以内（RM-002 / RM-001b）
    assumed_loss = (t.entry_price - t.stop_price) * t.shares * 2
    assert assumed_loss <= 50_000 * st06.risk_pct / 100 * 1.10


def test_r_multiple_arithmetic(st06):
    df = flat_then_breakout_then_crash()
    res = run(st06, {"TEST": df}, index=_index_for(df),
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0.0))
    for t in res.trades:
        if t.exit_price is None:
            continue
        risk = (t.entry_price - t.stop_price) * t.shares
        assert t.r_multiple == pytest.approx(t.pnl / risk)


def test_no_entry_when_market_regime_is_down(st06):
    """TOPIX が200日線を下回っている間は買わない（DEC-060 / common.yaml）。"""
    df = flat_then_breakout()
    idx = _index_for(df)
    falling = idx.copy()
    falling["close"] = idx["close"].iloc[0] - (idx["close"] - idx["close"].iloc[0])
    res = run(st06, {"TEST": df}, index=falling,
              config=BacktestConfig(initial_equity=1_000_000))
    assert not res.trades


def test_metrics_shape(st06):
    df = flat_then_breakout_then_crash()
    res = run(st06, {"TEST": df}, index=_index_for(df),
              config=BacktestConfig(initial_equity=1_000_000))
    m = res.metrics()
    assert m["取引数"] >= 1
    assert 0 <= m["勝率"] <= 100
    assert m["最大DD"] <= 0


def test_metrics_report_worst_trade():
    """最悪トレードR と最大単一損失（決算ギャップ等の尾部リスクを見る）。"""
    from pathlib import Path

    from short_trade.backtest import BacktestConfig, run
    from short_trade.spec import load_all
    from synthetic import flat_then_breakout_then_crash, rising_index

    spec = next(s for s in load_all(Path(__file__).resolve().parents[1] / "catalog") if s.id == "ST-06")
    df = flat_then_breakout_then_crash()
    idx = rising_index(len(df), start=str(df.index[0].date()))
    res = run(spec, {"T": df}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1, max_position_pct=100))
    m = res.metrics()
    assert "最悪トレードR" in m and "最大単一損失" in m
    closed = [t for t in res.trades if not t.forced]
    assert m["最大単一損失"] == min(t.pnl for t in closed)           # 負けが無ければ最小の勝ちになる
    assert m["最悪トレードR"] == min(t.r_multiple for t in closed)
