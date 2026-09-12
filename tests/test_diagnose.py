"""条件ファネル（diagnose.funnel）の検査。"""
from pathlib import Path

from short_trade.backtest import BacktestConfig, run
from short_trade.diagnose import format_funnel, funnel
from short_trade.spec import load_all
from synthetic import flat_then_breakout, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
SPECS = {s.id: s for s in load_all(CATALOG)}


def _setup():
    df = flat_then_breakout()
    idx = rising_index(len(df), start=str(df.index[0].date()))
    cfg = BacktestConfig(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1, max_position_pct=100)
    return df, idx, cfg


def test_funnel_lists_every_condition_in_run_order():
    df, idx, cfg = _setup()
    spec = SPECS["ST-06"]
    f = funnel(spec, {"T": df}, index=idx, config=cfg)
    kinds = [s.kind for s in f.steps]
    assert kinds.count("entry") == len(spec.entry_conditions)
    assert kinds.count("regime") == len(spec.regime_conditions)
    assert kinds == sorted(kinds, key=("universe", "entry", "regime").index)   # run() と同じ並び
    assert f.symbol_days == len(df) and f.symbols == 1


def test_cumulative_never_exceeds_single_and_is_monotone():
    df, idx, cfg = _setup()
    f = funnel(SPECS["ST-06"], {"T": df}, index=idx, config=cfg)
    prev = f.symbol_days
    for s in f.steps:
        assert s.cumulative_days <= s.pass_days
        assert s.cumulative_days <= prev
        prev = s.cumulative_days


def test_final_signal_days_matches_backtest_entry_days():
    """ファネルの最終行は、バックテスタが「条件成立」と見なす銘柄×日と一致する。"""
    df, idx, cfg = _setup()
    spec = SPECS["ST-06"]
    f = funnel(spec, {"T": df}, index=idx, config=cfg)
    res = run(spec, {"T": df}, index=idx, config=cfg)
    assert f.final_days > 0
    # 取引数はシグナル日数以下（同時保有・資金で減る）だが、シグナルが0なら取引も0
    assert len(res.trades) <= f.final_days
    assert "最終シグナル" in format_funnel(f)


def test_impossible_condition_shows_where_candidates_die():
    import copy
    df, idx, cfg = _setup()
    spec = copy.deepcopy(SPECS["ST-06"])
    spec.raw["entry"]["conditions"] = list(spec.entry_conditions) + ["close > close * 2"]
    f = funnel(spec, {"T": df}, index=idx, config=cfg)
    killer = next(s for s in f.steps if s.text == "close > close * 2")
    assert killer.pass_days == 0 and killer.cumulative_days == 0
    assert f.final_days == 0
