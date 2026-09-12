"""敵対的レビュー（docs/21）で見つけた不具合の回帰テスト。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_trade.backtest import BacktestConfig, run
from short_trade.indicators import build_namespace, evaluate
from short_trade.spec import StrategySpec, load_all, load_common
from synthetic import flat_then_breakout, flat_then_breakout_then_crash, make_bars, rising_index

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
COMMON = load_common(CATALOG)
SPECS = {s.id: s for s in load_all(CATALOG)}


def _spec(**overrides) -> StrategySpec:
    """テスト用の最小仕様。ドンチャン風の入り方に、上書きした exit/sizing を載せる。"""
    raw = {
        "id": "TEST", "name": "test", "factor": "F1", "hypothesis": "t",
        "regime_filter": ["INDEX.close > SMA(INDEX.close, 200)"],
        "entry": {"conditions": ["close > HIGHEST(high, 20)[-1]"]},
        "exit": {"stop": {"initial": "LOWEST(low, 10)"}, "rules": []},
        "sizing": {"risk_pct": 1.0},
        "params_to_optimize": [], "deviations": [], "expected_correlation": {}, "data_required": [],
    }
    for k, v in overrides.items():
        raw[k] = v
    return StrategySpec(id="TEST", name="test", factor="F1", phase=2, common=COMMON, raw=raw)


def _idx(df):
    return rising_index(len(df), start=str(df.index[0].date()))


def _cfg(**kw):
    base = dict(initial_equity=1_000_000, slippage_pct=0.0, hysteresis_days=1)
    base.update(kw)
    return BacktestConfig(**base)


# ---- 不具合1: `holding_days >= 20 and unrealized_r < 1.0` が int() で落ちていた ----
def test_combined_position_rule_parses_and_fires():
    spec = _spec(exit={"stop": {"initial": "close_at_entry * 0.5"},   # 損切りは事実上効かない
                       "rules": [{"name": "時間手仕舞い",
                                  "condition": "holding_days >= 5 and unrealized_r < 100"}]})
    df = flat_then_breakout()
    res = run(spec, {"T": df}, index=_idx(df), config=_cfg())
    assert res.trades and res.trades[0].exit_reason == "時間手仕舞い"


# ---- 不具合2: `unrealized_r >= 2.0` 単独の手仕舞い条件が一度も発火しなかった ----
def test_unrealized_r_only_rule_fires():
    spec = _spec(exit={"stop": {"initial": "close_at_entry * 0.98"},
                       "rules": [{"name": "目標到達", "condition": "unrealized_r >= 2.0"}]})
    df = flat_then_breakout(n_up=60, step=3.0)
    res = run(spec, {"T": df}, index=_idx(df), config=_cfg())
    reasons = {t.exit_reason for t in res.trades}
    assert "目標到達" in reasons, reasons


# ---- 不具合3: データが途中で終わる銘柄の建玉が消えて資産から抜け落ちていた ----
def test_delisted_symbol_is_force_closed_and_cash_returns():
    spec = _spec()
    long_df = flat_then_breakout(n_flat=60, n_up=80)
    short_df = flat_then_breakout(n_flat=60, n_up=20)          # 途中で上場廃止したことにする
    res = run(spec, {"LONG": long_df, "SHORT": short_df}, index=_idx(long_df), config=_cfg())
    short_trades = [t for t in res.trades if t.symbol == "SHORT"]
    assert short_trades and short_trades[0].exit_reason == "データ終了"
    assert short_trades[0].exit_date == short_df.index[-1]
    # 資産曲線が「消えた建玉」ぶん急落していないこと
    eq = res.equity_curve
    assert (eq.pct_change().dropna() > -0.05).all()


# ---- 不具合4: 寄付でギャップアップすると、実リスクがリスク予算を超えていた ----
def test_shares_are_resized_at_fill_when_gap_up():
    spec = _spec()
    df = flat_then_breakout()
    # ブレイク翌日の始値を大きく飛ばす（損切り幅が広がる）
    sig = df.index[60]                      # 最初のブレイク日
    nxt = df.index[61]
    df.loc[nxt, "open"] = df.at[sig, "close"] * 1.10
    df.loc[nxt, "high"] = max(df.at[nxt, "high"], df.at[nxt, "open"] * 1.01)
    cfg = _cfg(initial_equity=50_000, max_position_pct=100.0, assumed_loss_multiple=2.0)
    res = run(spec, {"T": df}, index=_idx(df), config=cfg)
    t = res.trades[0]
    assumed_loss = (t.entry_price - t.stop_price) * t.shares * 2.0
    assert assumed_loss <= 50_000 * 0.01 + 1e-6


# ---- 不具合5: レジームのヒステリシス（3日連続）が実装されていなかった ----
def _index_up_for_only(df, n_days: int):
    """常に200日線の下にある指数を作り、上昇局面の中の n_days 日だけ線の上に出す。"""
    idx = _idx(df)
    falling = idx.copy()
    falling["close"] = np.linspace(2000.0, 1000.0, len(idx))       # 一本調子で下落 → 常に SMA200 の下
    window = idx.index[(idx.index >= df.index[70])][:n_days]        # 上昇局面（毎日が新高値）の中
    falling.loc[window, "close"] = 5000.0                            # 2〜3日だけ大きく上に出す
    return falling


def test_regime_hysteresis_blocks_two_day_flip_but_allows_three():
    spec = _spec()
    df = flat_then_breakout()
    two = run(spec, {"T": df}, index=_index_up_for_only(df, 2), config=_cfg(hysteresis_days=3))
    three = run(spec, {"T": df}, index=_index_up_for_only(df, 3), config=_cfg(hysteresis_days=3))
    assert not two.trades, "2日しか成立していないのに建てている"
    assert three.trades, "3日連続で成立したのに建てていない"


# ---- 不具合6: ST-09 の月次リバランスが無視され、毎日エントリーしていた ----
def test_momentum_enters_only_on_month_end():
    n = 400
    strong = make_bars([100 * (1.004 ** i) for i in range(n)])
    weak = make_bars([100 * (0.999 ** i) for i in range(n)])
    idx = rising_index(n, start=str(strong.index[0].date()))
    res = run(SPECS["ST-09"], {"S": strong, "W": weak}, index=idx, config=_cfg())
    assert res.trades
    for t in res.trades:
        decision = strong.index[strong.index.get_loc(t.entry_date) - 1]
        month_end = strong.index[strong.index.to_period("M") == decision.to_period("M")][-1]
        assert decision == month_end, f"月末以外にエントリーしている: {decision.date()}"


# ---- 不具合7: 式の max/min が Python 組み込みで Series を比較できなかった ----
def test_elementwise_max_in_expression():
    df = make_bars([100, 110, 90, 120])
    out = evaluate("max(close, 100)", build_namespace(df))
    assert list(out) == [100, 110, 100, 120]


# ---- 不具合8: 強制手仕舞い（期末評価）が勝率・PFに混ざっていた ----
def test_forced_exits_are_excluded_from_win_rate():
    spec = _spec()
    df = flat_then_breakout()
    res = run(spec, {"T": df}, index=_idx(df), config=_cfg())
    m = res.metrics()
    assert m["強制手仕舞い"] == 1 and m["取引数"] == 0


# ---- 不具合9: 指数データが途切れると、最後のレジーム判定を無期限に引きずっていた ----
def test_stale_index_does_not_keep_regime_true():
    spec = _spec()
    df = flat_then_breakout()
    idx = _idx(df)
    truncated = idx.loc[: df.index[45]]            # ブレイク（60日目）の15営業日前で指数が止まる（猶予5日を超える）
    res = run(spec, {"T": df}, index=truncated, config=_cfg())
    assert not res.trades, "指数が止まっているのに建てている（fail-closed になっていない）"
    assert any("指数データが" in w for w in res.warnings)


# ---- 第2パス ----
def test_invalid_bars_are_rejected_before_any_signal():
    """FR-108: 高値<安値や価格<=0 のデータで黙ってシグナルを出さない。"""
    spec = _spec()
    df = flat_then_breakout()
    bad = df.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = -1.0
    with pytest.raises(ValueError, match="異常値"):
        run(spec, {"T": bad}, index=_idx(df), config=_cfg())


def test_strategy_specific_max_positions_is_enforced():
    """ST-09 の definitions.max_positions=5 が共通の8本より優先される。"""
    n = 400
    data = {f"S{i}": make_bars([100 * ((1.004 + i * 0.0003) ** k) for k in range(n)]) for i in range(12)}
    idx = rising_index(n, start=str(data["S0"].index[0].date()))
    res = run(SPECS["ST-09"], data, index=idx, config=_cfg(initial_equity=10_000_000, max_position_pct=100))
    # どの時点でも同時保有は5本以下
    open_count = {}
    for t in res.trades:
        for d in pd.bdate_range(t.entry_date, t.exit_date):
            open_count[d] = open_count.get(d, 0) + 1
    assert max(open_count.values()) <= 5


def test_daily_loss_limit_blocks_new_entries_that_day():
    """RM-020: 当日の純損失が上限に達したら、その日は新規建てを決めない。"""
    spec = _spec()
    # 2銘柄: A は保有中に急落して当日 -3% を作る、B は同日にブレイクする
    a = flat_then_breakout_then_crash(n_flat=60, n_up=5, n_down=40, drop=6.0)
    b = flat_then_breakout(n_flat=66, n_up=40)
    idx = _idx(b)
    limited = run(spec, {"A": a, "B": b}, index=idx,
                  config=_cfg(initial_equity=100_000, max_position_pct=100, daily_loss_limit_pct=0.5))
    unlimited = run(spec, {"A": a, "B": b}, index=idx,
                    config=_cfg(initial_equity=100_000, max_position_pct=100, daily_loss_limit_pct=None))
    assert "日次損失上限に到達（当日の新規建て停止）" in limited.rejections
    assert "日次損失上限に到達（当日の新規建て停止）" not in unlimited.rejections


def test_drawdown_derisk_shrinks_position_size():
    """RM-022: ドローダウン中は新規建てのサイズが縮む。"""
    spec = _spec()
    a = flat_then_breakout_then_crash(n_flat=60, n_up=5, n_down=40, drop=6.0)   # 先に損を作る
    b = flat_then_breakout(n_flat=110, n_up=30)                                   # DD中にブレイク
    idx = _idx(b)
    base = run(spec, {"A": a, "B": b}, index=idx, config=_cfg(initial_equity=100_000, max_position_pct=100))
    derisk = run(spec, {"A": a, "B": b}, index=idx,
                 config=_cfg(initial_equity=100_000, max_position_pct=100, drawdown_derisk=[(1.0, 0.25)]))
    b_base = [t for t in base.trades if t.symbol == "B"]
    b_derisk = [t for t in derisk.trades if t.symbol == "B"]
    assert b_base and b_derisk
    assert b_derisk[0].shares < b_base[0].shares


def test_universe_turnover_filter_excludes_thin_stocks():
    spec = _spec()
    thin = flat_then_breakout()
    thin["volume"] = 10.0                                     # 売買代金 ≈ 1,000円/日
    res = run(spec, {"T": thin}, index=_idx(thin), config=_cfg(min_avg_turnover_20d=50_000_000))
    assert not res.trades


# ---- 第3パス: entry.rank が説明文のままで、候補を銘柄コード順に採用していた ----
def test_candidates_are_taken_in_rank_order_not_symbol_order():
    n = 400
    # 銘柄コードは "A"(弱) < "B"(中) < "Z"(強)。強い順に採るなら Z が選ばれるはず
    data = {
        "A": make_bars([100 * (1.0030 ** k) for k in range(n)]),
        "B": make_bars([100 * (1.0035 ** k) for k in range(n)]),
        "Z": make_bars([100 * (1.0060 ** k) for k in range(n)]),
    }
    idx = rising_index(n, start=str(data["A"].index[0].date()))
    res = run(SPECS["ST-09"], data, index=idx,
              config=_cfg(initial_equity=10_000_000, max_position_pct=100, max_positions=1))
    assert res.trades and res.trades[0].symbol == "Z", [t.symbol for t in res.trades]


def test_prose_rank_is_rejected_as_spec_error():
    from short_trade.spec import SpecError
    spec = _spec(entry={"conditions": ["close > HIGHEST(high, 20)[-1]"], "rank": "高い順"})
    with pytest.raises(SpecError):
        _ = spec.rank


# ---------------------------------------------------------------- 決算跨ぎ方針（RM-001d, docs/28 §28.1）

def _earnings_setup():
    from synthetic import flat_then_breakout, rising_index
    df = flat_then_breakout(n_flat=60, n_up=40)
    idx = rising_index(len(df), start=str(df.index[0].date()))
    spec = next(s for s in load_all(CATALOG) if s.id == "ST-06")
    # 上昇の途中（ブレイクの 10 営業日後）に決算日を置く
    breakout = df.index[60]
    earn = df.index[70]
    return spec, df, idx, breakout, earn


def _run_policy(policy, cushion=1.0):
    spec, df, idx, breakout, earn = _earnings_setup()
    cfg = BacktestConfig(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1, max_position_pct=100,
                         earnings_dates={"T": [earn]}, earnings_policy=policy, earnings_cushion_r=cushion)
    return run(spec, {"T": df}, index=idx, config=cfg), earn


def test_policy_exit_closes_before_earnings():
    res, earn = _run_policy("exit")
    assert res.trades and res.trades[0].exit_reason == "決算跨ぎ回避"
    assert res.trades[0].exit_date < earn


def test_policy_entry_only_holds_through_earnings():
    res, earn = _run_policy("entry_only")
    assert res.trades
    assert all(t.exit_reason != "決算跨ぎ回避" for t in res.trades)
    assert res.trades[0].exit_date > earn


def test_policy_cushion_depends_on_unrealized_r():
    held, earn = _run_policy("cushion", cushion=0.1)       # 直前で +0.1R 以上あれば持ち越す
    cut, _ = _run_policy("cushion", cushion=50.0)          # 50R は不可能 → 手仕舞う
    assert held.trades[0].exit_date > earn
    assert cut.trades[0].exit_reason == "決算跨ぎ回避"


def test_unknown_policy_is_rejected():
    spec, df, idx, _, earn = _earnings_setup()
    with pytest.raises(ValueError):
        run(spec, {"T": df}, index=idx,
            config=BacktestConfig(earnings_dates={"T": [earn]}, earnings_policy="hold"))
