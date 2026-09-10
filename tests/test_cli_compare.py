"""compare コマンドとリスク率上書きの検査。キャッシュは合成データで作る。"""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_trade import cli
from short_trade.backtest import BacktestConfig, run
from short_trade.spec import load_all
from synthetic import flat_then_breakout, make_bars, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"


def _v2_frame(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": bars.index.strftime("%Y-%m-%d"), "Code": "00000",
        "O": bars["open"], "H": bars["high"], "L": bars["low"], "C": bars["close"], "Vo": bars["volume"],
        "AdjO": bars["open"], "AdjH": bars["high"], "AdjL": bars["low"], "AdjC": bars["close"], "AdjVo": bars["volume"],
    }).reset_index(drop=True)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    data_dir = tmp_path / "jquants"
    (data_dir / "daily").mkdir(parents=True)
    n = 400
    for i, code in enumerate(("1111", "2222", "3333")):
        bars = make_bars([1000 * (1.002 + i * 0.0005) ** k for k in range(n)])
        _v2_frame(bars).to_parquet(data_dir / "daily" / f"{code}.parquet")
    idx = rising_index(n, start="2024-01-01")
    idx.to_parquet(data_dir / "index.parquet")
    monkeypatch.setattr(cli, "DATA", data_dir)
    monkeypatch.setattr(cli, "EARNINGS_PATH", data_dir / "earnings.parquet")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    return data_dir


def test_risk_pct_override_changes_position_size():
    spec = next(s for s in load_all(CATALOG) if s.id == "ST-06")
    df = flat_then_breakout()
    idx = rising_index(len(df), start=str(df.index[0].date()))
    one = run(spec, {"T": df}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1,
                                    max_position_pct=100, risk_pct_override=1.0))
    two = run(spec, {"T": df}, index=idx,
              config=BacktestConfig(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1,
                                    max_position_pct=100, risk_pct_override=2.0))
    assert one.trades and two.trades
    assert two.trades[0].shares == pytest.approx(one.trades[0].shares * 2, rel=0.02)


def test_compare_writes_report_with_all_combinations(cache, capsys):
    args = Namespace(strategy="ST-06", risk="1.0,2.0", slippage_levels="0.0,0.1",
                     equity=1_000_000, start=None, end=None)
    assert cli.cmd_compare(args) == 0
    report = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())
    rows = report["results"]["ST-06"]
    assert len(rows) == 4
    assert {(r["risk_pct"], r["slippage_pct"]) for r in rows} == {(1.0, 0.0), (1.0, 0.1), (2.0, 0.0), (2.0, 0.1)}
    assert report["earnings_applied"] is False
    out = capsys.readouterr().out
    assert "リスク%" in out and "compare_report.json" in out


def test_compare_applies_earnings_when_present(cache):
    pd.DataFrame({"PubDate": ["2024-06-01"], "SchDate": ["2024-06-10"], "Code": ["11110"]}).to_parquet(
        cli.EARNINGS_PATH)
    args = Namespace(strategy="ST-06", risk="1.0", slippage_levels="0.0",
                     equity=1_000_000, start=None, end=None)
    cli.cmd_compare(args)
    report = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())
    assert report["earnings_applied"] is True
