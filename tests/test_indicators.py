import numpy as np
import pandas as pd
import pytest

from short_trade.indicators import (
    ATR, HIGHEST, LOWEST, PCTRANK, RSI, SMA, expand_shifts, evaluate, build_namespace,
)
from synthetic import make_bars


def s(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_sma_uses_only_past_and_current():
    x = s([1, 2, 3, 4, 5])
    out = SMA(x, 3)
    assert np.isnan(out.iloc[1])           # 窓が埋まるまでは NaN
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_highest_lowest_include_current_bar():
    x = s([5, 3, 9, 1, 7])
    assert HIGHEST(x, 3).iloc[2] == 9
    assert LOWEST(x, 3).iloc[3] == 1


def test_rsi_bounds_and_direction():
    up = RSI(s(list(range(1, 40))), 2)
    down = RSI(s(list(range(40, 1, -1))), 2)
    assert up.dropna().min() > 60          # 上げ続ければ高い
    assert down.dropna().max() < 40        # 下げ続ければ低い
    both = pd.concat([up, down]).dropna()
    assert both.between(0, 100).all()


def test_pctrank_is_percentile():
    x = s([1, 2, 3, 4, 100])
    assert PCTRANK(x, 5).iloc[4] == pytest.approx(100.0)


def test_atr_positive():
    df = make_bars([100 + i for i in range(40)])
    atr = ATR(14, high=df["high"], low=df["low"], close=df["close"])
    assert atr.dropna().gt(0).all()


@pytest.mark.parametrize("expr,expected", [
    ("close[-1]", "SHIFT(close, 1)"),
    ("HIGHEST(high, 20)[-1]", "SHIFT(HIGHEST(high, 20), 1)"),
    ("SMA(close, 200) > SMA(close, 200)[-20]",
     "SMA(close, 200) > SHIFT(SMA(close, 200), 20)"),
    ("INDEX.close[-3]", "SHIFT(INDEX.close, 3)"),
    ("close > close[-1] and low[-2] > 0", "close > SHIFT(close, 1) and SHIFT(low, 2) > 0"),
])
def test_expand_shifts(expr, expected):
    assert expand_shifts(expr) == expected


def test_evaluate_breakout_condition():
    df = make_bars([100] * 25 + [130])
    ns = build_namespace(df)
    out = evaluate("close > HIGHEST(high, 20)[-1]", ns).fillna(False)
    assert not out.iloc[:25].any()
    assert bool(out.iloc[25])


@pytest.mark.parametrize("expr,expected", [
    ("a > 1 and b < 2", "(a > 1) & (b < 2)"),
    ("a > 1 and b < 2 and c == 3", "(a > 1) & (b < 2) & (c == 3)"),
    ("a > 1 or b < 2", "(a > 1) | (b < 2)"),
    ("max(a, b) > 1 and c < 2", "(max(a, b) > 1) & (c < 2)"),
    ("a > 1", "a > 1"),
])
def test_normalize_logical(expr, expected):
    from short_trade.indicators import normalize_logical
    assert normalize_logical(expr) == expected


def test_evaluate_handles_and_between_series():
    df = make_bars([100] * 30 + [130] * 5)
    ns = build_namespace(df)
    out = evaluate("close > SMA(close, 20) and volume > 0", ns).fillna(False)
    assert bool(out.iloc[-1])
