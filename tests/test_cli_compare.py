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


def test_fetch_index_path_defines_start(monkeypatch, tmp_path):
    """`fetch --index` で start が未定義になる不具合の回帰テスト。"""
    from short_trade import cli as _cli

    calls = {}

    class Fake:
        def coverage(self):
            return ("2016-09-10", None)

        def topix(self, *, start=None, end=None):
            calls["start"] = start
            return pd.DataFrame({"Date": ["2024-01-04", "2024-01-05"], "O": [1, 1], "H": [1, 1], "L": [1, 1], "C": [1000.0, 1001.0]})

    import short_trade.jquants as jq
    monkeypatch.setattr(jq, "JQuantsClient", lambda *a, **k: Fake())
    monkeypatch.setattr(_cli, "ROOT", tmp_path)
    monkeypatch.setattr(_cli, "DATA", tmp_path / "data" / "jquants")
    args = Namespace(index="topix", start=None, end=None, codes=None, refresh=False,
                     earnings=False, universe=False, market="", top=10)
    assert _cli.cmd_fetch(args) == 0
    assert calls["start"] == "2016-09-10"
    assert (tmp_path / "data" / "jquants" / "index.parquet").exists()


def test_compare_defaults_to_all_phase2_strategies(cache):
    args = Namespace(strategy=None, phase=2, risk="1.0", slippage_levels="0.0",
                     equity=1_000_000, start=None, end=None)
    assert cli.cmd_compare(args) == 0
    report = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())
    assert {"ST-04", "ST-06", "ST-09", "ST-12"} <= set(report["results"])
    assert all(len(rows) == 1 for rows in report["results"].values())


def test_funnel_command_writes_report(cache, capsys):
    args = Namespace(strategy="ST-06", phase=None, start=None, end=None)
    assert cli.cmd_funnel(args) == 0
    report = json.loads((cli.ROOT / "data" / "funnel_report.json").read_text())
    f = report["funnels"]["ST-06"]
    assert f["symbols"] == 3 and f["steps"] and "final_signal_days" in f
    assert "ここまでAND" in capsys.readouterr().out


def test_compare_adds_earnings_policy_rows_only_when_earnings_exist(cache):
    args = Namespace(strategy="ST-06", phase=None, risk="1.0", slippage_levels="0.0,0.1",
                     earnings_policies="entry_only,cushion", equity=1_000_000, start=None, end=None)
    cli.cmd_compare(args)
    rows = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())["results"]["ST-06"]
    default_policy = load_all(CATALOG)[0].common["risk"]["earnings_policy"]
    assert len(rows) == 2 and all(r["earnings_policy"] == default_policy for r in rows)   # 決算日なし → 方針の行は出ない

    pd.DataFrame({"PubDate": ["2024-06-01"], "SchDate": ["2024-06-10"], "Code": ["11110"]}).to_parquet(cli.EARNINGS_PATH)
    cli.cmd_compare(args)
    rows = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())["results"]["ST-06"]
    assert len(rows) == 4
    assert [(r["slippage_pct"], r["earnings_policy"]) for r in rows] == [
        (0.0, default_policy), (0.1, default_policy), (0.1, "entry_only"), (0.1, "cushion")]


def test_detector_memo_returns_same_frame_for_same_bars():
    from short_trade.backtest import _DETECTOR_MEMO, _bind_detectors
    from synthetic import vcp_pattern
    spec = next(s for s in load_all(CATALOG) if s.id == "ST-04")
    bars, _ = vcp_pattern()
    _DETECTOR_MEMO.clear()
    ns1, ns2 = {}, {}
    _bind_detectors(spec, ns1, bars)
    _bind_detectors(spec, ns2, bars)
    assert len(_DETECTOR_MEMO) == 1
    # 足が変われば別のキーになる
    other = bars.copy(); other.loc[other.index[-1], "close"] *= 1.01
    _bind_detectors(spec, {}, other)
    assert len(_DETECTOR_MEMO) == 2
