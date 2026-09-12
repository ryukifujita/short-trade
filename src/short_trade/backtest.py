"""イベント駆動バックテスタ。

設計の核（docs/18 §18.1）:
  - すべての戦略は「当日終値で判定 → 翌営業日9:00の成行」で執行する。
    判定日 t で参照できるのは t までの確定足だけであり、約定価格は t+1 の始値である。
  - S株は逆指値が置けないため、損切りは論理ストップ＋翌執行枠の成行（RM-001a）。
  - 想定損失は損切り幅の assumed_loss_multiple 倍で見積もる（RM-001b）。

レビューで直した点（docs/21）:
  - 約定価格が確定してからサイズを再計算する。ギャップで実リスクが予算を超えないようにする
  - データが途切れた銘柄（上場廃止など）は最終値で強制手仕舞いし、資産が消えないようにする
  - ポジション単位の条件（holding_days / unrealized_r）は式として評価する
  - トレーリングの発動条件（+2R 等）、月次リバランス、レジームのヒステリシスを実装
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .detectors import DETECTORS
from .indicators import _Frame, build_namespace, evaluate
from .spec import StrategySpec

# ポジション単位でしか評価できない名前（式ではなく Python 側で判定する）
_POSITION_SCOPED = ("holding_days", "unrealized_r")
_FORCED_EXITS = ("期末評価", "データ終了")

_XRANK_RE = re.compile(r"PCTRANK\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*universe\s*\)")
_R_RE = re.compile(r"^\+?\s*([0-9.]+)\s*R$")


class UnsupportedSpec(RuntimeError):
    """仕様が参照している要素がまだ実装されていない。

    黙って無視すると「シグナルが出ない戦略」に見えてしまい、検証結果を誤読する。
    未実装は未実装として、名前を挙げて落とす。
    """


# ---------------------------------------------------------------- 補助

def _xrank_name(inner: str) -> str:
    return f"__XRANK_{inner}"


def _rewrite_xrank(expr: str) -> str:
    """`PCTRANK(x, universe)` を事前計算済みの名前に置き換える。"""
    return _XRANK_RE.sub(lambda m: _xrank_name(m.group(1)), expr)


def _xrank_targets(spec: StrategySpec) -> list[str]:
    texts = list(spec.entry_conditions) + list(spec.regime_conditions)
    texts += [r["condition"] for r in spec.exit_rules]
    texts += [spec.stop_initial] + ([spec.stop_trailing] if spec.stop_trailing else [])
    names: list[str] = []
    for t in texts:
        names += _XRANK_RE.findall(t)
    return list(dict.fromkeys(names))


def _eval_or_unsupported(spec_id: str, what: str, expr: str, namespace: dict):
    """未実装の参照は UnsupportedSpec に変換し、それ以外の例外はそのまま通す。

    以前は全例外を握りつぶしていたため、仕様のタイプミスが「損切り価格が不正」という
    却下件数にしか見えず、バグを隠していた。
    """
    try:
        return evaluate(expr, namespace)
    except (NameError, AttributeError) as e:
        raise UnsupportedSpec(f"{spec_id}: {what} `{expr}` が未実装の要素を参照しています: {e}") from e
    except SyntaxError as e:
        raise UnsupportedSpec(f"{spec_id}: {what} が式になっていません: {expr[:60]}") from e


def _stop_at_fill(spec_id: str, expr: str, namespace: dict, decision_date, fill_price: float) -> float:
    """約定価格が確定したあとに損切り価格を確定させる。"""
    ns = dict(namespace, close_at_entry=fill_price)
    series = _eval_or_unsupported(spec_id, "損切り式", expr, ns)
    if isinstance(series, pd.Series):
        return float(series.get(decision_date, np.nan))
    return float(series)


_ALIGN_FFILL_LIMIT = 5   # 営業日。指数の欠落がこれを超えたら「判定不能」＝False にする


def _align(series: pd.Series, target: pd.Index) -> pd.Series:
    """条件式の評価結果を銘柄の日付に整列する。

    指数（INDEX）を参照する条件は指数側の日付で返るため、銘柄の営業日へ寄せる。
    直近の判定は数日なら引き継ぐが、指数データが途切れた場合に最後の状態を
    無期限に引きずると、TOPIX の取得が止まっただけで買い続ける事故になる。
    fail-closed（RM-034）として、欠落が _ALIGN_FFILL_LIMIT を超えたら False にする。
    """
    if series.index.equals(target):
        return series.fillna(False).astype(bool)
    return (series.reindex(series.index.union(target))
                  .ffill(limit=_ALIGN_FFILL_LIMIT)
                  .reindex(target).fillna(False).astype(bool))


def _with_hysteresis(flag: pd.Series, days: int) -> pd.Series:
    """DEC-012: 条件が days 営業日連続で成立して初めて True にする。"""
    if days <= 1:
        return flag
    return flag.astype(int).rolling(days, min_periods=days).min().fillna(0).astype(bool)


def _position_rule(cond: str, *, holding_days: int, unrealized_r: float) -> bool:
    """`holding_days >= 20 and unrealized_r < 1.0` のような式をスカラーで評価する。"""
    return bool(eval(cond, {"__builtins__": {}},  # noqa: S307 - 自前の仕様ファイルのみ
                     {"holding_days": holding_days, "unrealized_r": unrealized_r}))


def _parse_r(text: str | None) -> float | None:
    if not text:
        return None
    m = _R_RE.match(str(text).strip())
    if not m:
        raise UnsupportedSpec(f"trailing_activates_at の書式が不正です: {text!r}（例: '+2R'）")
    return float(m.group(1))


def _month_end_dates(dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
    """各月の最終営業日（データに存在する日付のうち）。"""
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    return set(pd.Timestamp(v) for v in s.groupby(s.index.to_period("M")).max().values)


# ---------------------------------------------------------------- データ構造

@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""

    @property
    def risk_per_share(self) -> float:
        return self.entry_price - self.stop_price

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def r_multiple(self) -> float:
        """MM-07: 損益を初期リスク1個ぶんで割った値。"""
        risk = self.risk_per_share * self.shares
        return self.pnl / risk if risk > 0 else 0.0

    @property
    def forced(self) -> bool:
        return self.exit_reason in _FORCED_EXITS


@dataclass
class Position:
    trade: Trade
    stop_price: float
    bars_held: int = 0
    last_close: float = 0.0
    trailing_active: bool = False
    pending_exit: str | None = None    # 翌営業日の始値で手仕舞う理由


@dataclass
class _PendingEntry:
    symbol: str
    stop_price: float
    shares: int
    decision_date: pd.Timestamp
    risk_budget: float      # 想定損失の上限（円）
    value_cap: float        # 投下額の上限（円）


@dataclass
class BacktestConfig:
    initial_equity: float = 50_000.0
    slippage_pct: float = 0.10          # 片道。成行のため保守的に置く（VR-011）
    commission_pct: float = 0.0         # SBI証券ゼロ革命（PL-002 / CR-041）
    max_positions: int = 8
    max_position_pct: float = 15.0
    max_portfolio_heat_pct: float = 6.0
    assumed_loss_multiple: float = 2.0
    hysteresis_days: int = 3
    earnings_dates: dict[str, list[pd.Timestamp]] | None = None   # 未指定なら決算跨ぎ禁止は無効
    earnings_blackout_days: int = 1
    # RM-001d の運用方針（docs/28 §28.1）。
    #   exit       … 決算前に必ず手仕舞う（原案）。エントリーも前後 k 日は禁止
    #   entry_only … エントリー禁止だけ。保有中の玉は決算をまたいで持つ
    #   cushion    … 含み益が earnings_cushion_r 以上なら持ち越し、未満なら手仕舞う
    earnings_policy: str = "exit"
    earnings_cushion_r: float = 1.0
    daily_loss_limit_pct: float | None = 2.0                       # RM-020
    drawdown_derisk: list[tuple[float, float]] = field(default_factory=list)   # RM-022 [(dd%, 倍率)]
    min_avg_turnover_20d: float = 0.0                              # ユニバース: 20日平均売買代金
    min_listed_days: int = 0                                       # ユニバース: 上場後の営業日数
    risk_pct_override: float | None = None                         # None なら戦略仕様の risk_pct を使う

    @classmethod
    def from_common(cls, common: dict[str, Any], **overrides: Any) -> "BacktestConfig":
        risk = common.get("risk", {})
        regime = common.get("regime", {})
        universe = common.get("universe", {})
        derisk = sorted(
            ((float(d["dd_pct"]), float(d["size_mult"])) for d in risk.get("drawdown_derisk", []) or []),
            reverse=True,
        )
        execution = common.get("execution", {})
        cfg = cls(
            slippage_pct=float(execution.get("slippage_pct", 0.10)),
            commission_pct=float(execution.get("commission_pct", 0.0)),
            max_positions=int(risk.get("max_concurrent_positions", 8)),
            max_position_pct=float(risk.get("max_position_pct", 15)),
            max_portfolio_heat_pct=float(risk.get("max_portfolio_heat_pct", 6.0)),
            assumed_loss_multiple=float(risk.get("assumed_loss_multiple", 2.0)),
            hysteresis_days=int(regime.get("hysteresis_days", 1)),
            earnings_blackout_days=int(risk.get("earnings_blackout_days", 1)),
            earnings_policy=str(risk.get("earnings_policy", "exit")),
            earnings_cushion_r=float(risk.get("earnings_cushion_r", 1.0)),
            daily_loss_limit_pct=risk.get("daily_loss_limit_pct"),
            drawdown_derisk=derisk,
            min_avg_turnover_20d=float(universe.get("min_avg_turnover_20d", 0) or 0),
            min_listed_days=int(universe.get("min_listed_days", 0) or 0),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    rejections: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        """docs/12 の KPI 定義に合わせる。強制手仕舞い（期末評価・データ終了）は勝率等から除く。"""
        closed = [t for t in self.trades if t.exit_price is not None and not t.forced]
        forced = [t for t in self.trades if t.forced]
        eq = self.equity_curve
        dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
        if not closed:
            return {"取引数": 0, "強制手仕舞い": len(forced), "最大DD": dd * 100}
        pnl = np.array([t.pnl for t in closed])
        r = np.array([t.r_multiple for t in closed])
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_loss = -losses.sum()
        return {
            "取引数": len(closed),
            "強制手仕舞い": len(forced),
            "勝率": float(len(wins) / len(closed) * 100),
            "総損益": float(pnl.sum() + sum(t.pnl for t in forced)),
            "平均利益": float(wins.mean()) if len(wins) else 0.0,
            "平均損失": float(losses.mean()) if len(losses) else 0.0,
            "プロフィットファクタ": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
            "平均R": float(r.mean()),
            "期待値": float(pnl.mean()),
            "平均保有日数": float(np.mean([t_bars for t_bars in self._bars_held(closed)])),
            "最大DD": dd * 100,
            "最大連敗": _max_streak(pnl <= 0),
        }

    def _bars_held(self, trades: list[Trade]) -> list[int]:
        idx = self.equity_curve.index
        out = []
        for t in trades:
            try:
                out.append(int(idx.get_loc(t.exit_date) - idx.get_loc(t.entry_date)))
            except KeyError:
                out.append(0)
        return out


def _max_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def validate_bars(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    """FR-108: 欠損・異常値・時刻逆転を検知する。壊れたデータでシグナルを出さない。"""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{sym}: 列が不足しています: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{sym}: 日付が単調増加ではありません")
    if df.index.has_duplicates:
        raise ValueError(f"{sym}: 日付が重複しています")
    clean = df.dropna(subset=required)
    bad = (clean["high"] < clean["low"]) | (clean["close"] <= 0) | (clean["open"] <= 0)
    if bad.any():
        raise ValueError(f"{sym}: 異常値があります（高値<安値、または価格<=0）: {list(clean.index[bad][:3])}")
    return clean


# ---------------------------------------------------------------- 本体

# ---------------------------------------------------------------- 名前空間の構築（run と diagnose で共用）

_DETECTOR_MEMO: dict[tuple, pd.DataFrame] = {}


def _detector_key(kind: str, params: dict, df: pd.DataFrame) -> tuple:
    """同じ足・同じ引数なら結果は同じ。compare のように同じデータで何度も走らせるときの再計算を避ける。"""
    return (kind, tuple(sorted(params.items())), len(df), str(df.index[0]), str(df.index[-1]),
            float(df["close"].iloc[-1]), float(df["volume"].sum()))


def _bind_detectors(spec: StrategySpec, ns: dict, df: pd.DataFrame) -> None:
    for name, cfg_d in spec.detector_definitions.items():
        kind = cfg_d["detector"]
        if kind not in DETECTORS:
            raise UnsupportedSpec(f"{spec.id}: 検出器 {kind!r} は登録されていません（detectors.py）")
        params = {k: v for k, v in cfg_d.items() if k not in ("detector", "note", "detection", "pivot", "v1_v2_v3")}
        key = _detector_key(kind, params, df)
        if key not in _DETECTOR_MEMO:
            if len(_DETECTOR_MEMO) > 2000:
                _DETECTOR_MEMO.clear()
            _DETECTOR_MEMO[key] = DETECTORS[kind](df, **params)
        ns[name] = _Frame(_DETECTOR_MEMO[key])


def _bind_definitions(spec: StrategySpec, ns: dict) -> None:
    for name, expr in spec.definitions.items():
        ns[name] = _eval_or_unsupported(spec.id, f"定義 {name}", expr, ns) if isinstance(expr, str) else expr


def cross_sectional_ranks(spec: StrategySpec, data: dict[str, pd.DataFrame],
                          index: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """`PCTRANK(x, universe)` の対象 x を全銘柄ぶん並べ、日ごとの百分位順位（0〜100）にする。"""
    xrank_names = _xrank_targets(spec)
    xranks: dict[str, pd.DataFrame] = {}
    if not xrank_names:
        return xranks
    raw_values: dict[str, dict[str, pd.Series]] = {n: {} for n in xrank_names}
    for sym, df in data.items():
        ns0 = build_namespace(df, index=index)
        _bind_detectors(spec, ns0, df)
        _bind_definitions(spec, ns0)
        for n in xrank_names:
            if isinstance(ns0.get(n), pd.Series):
                raw_values[n][sym] = ns0[n]
    for n in xrank_names:
        if not raw_values[n]:
            raise UnsupportedSpec(f"{spec.id}: クロスセクショナル順位の対象 {n} を計算できません")
        xranks[n] = pd.DataFrame(raw_values[n]).rank(axis=1, pct=True) * 100.0
    return xranks


def symbol_namespace(spec: StrategySpec, sym: str, df: pd.DataFrame, index: pd.DataFrame | None,
                     xranks: dict[str, pd.DataFrame]) -> dict:
    """1銘柄ぶんの式評価用の名前空間（価格列・指数・検出器・順位・定義）。"""
    ns = build_namespace(df, index=index)
    _bind_detectors(spec, ns, df)
    for n, frame in xranks.items():
        if sym in frame.columns:
            ns[_xrank_name(n)] = frame[sym].reindex(df.index)
    _bind_definitions(spec, ns)
    return ns


def universe_filters(cfg: "BacktestConfig", df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """common.universe 由来の銘柄フィルタ（名前つき）。"""
    out: list[tuple[str, pd.Series]] = []
    if cfg.min_avg_turnover_20d > 0:
        out.append((f"20日平均売買代金 >= {cfg.min_avg_turnover_20d:,.0f}",
                    ((df["close"] * df["volume"]).rolling(20).mean() >= cfg.min_avg_turnover_20d).fillna(False)))
    if cfg.min_listed_days > 0:
        out.append((f"上場後 {cfg.min_listed_days} 営業日以上",
                    pd.Series(np.arange(len(df)) >= cfg.min_listed_days, index=df.index)))
    return out


def regime_flag(spec: StrategySpec, cfg: "BacktestConfig", cond: str, ns: dict, target: pd.Index) -> pd.Series:
    flag = _align(_eval_or_unsupported(spec.id, "レジーム条件", _rewrite_xrank(cond), ns), target)
    return _with_hysteresis(flag, cfg.hysteresis_days)


def run(
    spec: StrategySpec,
    data: dict[str, pd.DataFrame],
    *,
    index: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    on_reject: Callable[[str], None] | None = None,
) -> BacktestResult:
    """1戦略のバックテストを実行する。

    data: {銘柄コード: 日付昇順の OHLCV DataFrame}。列は open/high/low/close/volume。
    index: 市場全体（TOPIX など）。式の `INDEX.close` に束縛される。
    """
    cfg = config or BacktestConfig.from_common(spec.common)
    if cfg.earnings_policy not in ("exit", "entry_only", "cushion"):
        raise ValueError(f"earnings_policy が不正です: {cfg.earnings_policy!r}（exit / entry_only / cushion）")
    warnings: list[str] = []
    if not data:
        raise ValueError("銘柄データが空です。fetch でデータを取得し、250本以上ある銘柄が存在することを確認してください")
    cleaned = {sym: validate_bars(sym, df) for sym, df in data.items()}
    for sym in data:
        dropped = len(data[sym]) - len(cleaned[sym])
        if dropped:
            warnings.append(f"{sym}: 欠損のある {dropped} 行を除外しました（FR-108）")
    data = cleaned
    max_positions = min(cfg.max_positions, spec.max_positions or cfg.max_positions)

    if spec.unsupported_definitions:
        raise UnsupportedSpec(
            f"{spec.id}: 専用の検出器が必要な定義があります: {', '.join(spec.unsupported_definitions)}"
        )
    if index is None and spec.regime_conditions:
        raise UnsupportedSpec(f"{spec.id}: レジーム条件があるのに指数データが渡されていません")
    if index is not None:
        last_stock = max(df.index[-1] for df in data.values())
        if index.index[-1] < last_stock:
            warnings.append(
                f"指数データが {index.index[-1].date()} で終わっており、銘柄データ（〜{last_stock.date()}）より短い。"
                f"欠落が{_ALIGN_FFILL_LIMIT}営業日を超える期間はレジーム判定不能として建てない"
            )
    if cfg.earnings_dates is None and spec.common.get("risk", {}).get("no_earnings_hold"):
        warnings.append("決算日データが無いため、決算跨ぎ禁止（RM-001d）は適用されていません")

    trailing_at = _parse_r(spec.raw.get("exit", {}).get("stop", {}).get("trailing_activates_at"))
    rebalance = spec.raw.get("rebalance")

    # --- 事前パス: クロスセクショナル順位 ---
    xranks = cross_sectional_ranks(spec, data, index)

    # --- 事前計算: 銘柄ごとの条件式 ---
    precomputed: dict[str, dict[str, Any]] = {}
    for sym, df in data.items():
        ns = symbol_namespace(spec, sym, df, index, xranks)
        entry = pd.Series(True, index=df.index)
        for _, flag in universe_filters(cfg, df):
            entry &= flag
        for cond in spec.entry_conditions:
            entry &= _align(_eval_or_unsupported(spec.id, "条件", _rewrite_xrank(cond), ns), df.index)
        for cond in spec.regime_conditions:
            entry &= regime_flag(spec, cfg, cond, ns, df.index)

        exits: dict[str, pd.Series] = {}
        for rule in spec.exit_rules:
            cond = rule["condition"]
            if any(k in cond for k in _POSITION_SCOPED):
                continue
            exits[rule["name"]] = _align(
                _eval_or_unsupported(spec.id, f"手仕舞い条件 {rule['name']}", _rewrite_xrank(cond), ns), df.index
            )

        rank_series = None
        if spec.rank is not None:
            rank_series = _eval_or_unsupported(spec.id, "rank", _rewrite_xrank(spec.rank[0]), ns)
        est_ns = dict(ns, close_at_entry=df["close"])   # 判定時の見積り。約定後に置き換える
        precomputed[sym] = {
            "entry": entry,
            "rank": rank_series,
            "stop_initial": _eval_or_unsupported(spec.id, "損切り式", _rewrite_xrank(spec.stop_initial), est_ns),
            "stop_trailing": (
                _eval_or_unsupported(spec.id, "トレーリング式", _rewrite_xrank(spec.stop_trailing), ns)
                if spec.stop_trailing else None
            ),
            "exits": exits,
            "ns": ns,
            "last_date": df.index[-1],
        }

    all_dates = sorted({d for df in data.values() for d in df.index})
    rebalance_days = _month_end_dates(all_dates) if rebalance else None
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    cash = cfg.initial_equity
    peak_equity = cfg.initial_equity
    prev_equity = cfg.initial_equity
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    rejections: dict[str, int] = {}
    pending: list[_PendingEntry] = []
    date_pos = {d: i for i, d in enumerate(all_dates)}

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1
        if on_reject:
            on_reject(reason)

    def in_earnings_blackout(sym: str, date: pd.Timestamp) -> bool:
        """決算日の前後 k 営業日。暦日で数えると週末をまたぐ発表を取りこぼす。"""
        if not cfg.earnings_dates or sym not in cfg.earnings_dates:
            return False
        k, here = cfg.earnings_blackout_days, date_pos.get(date)
        if here is None:
            return False
        for e in cfg.earnings_dates[sym]:
            ep = date_pos.get(pd.Timestamp(e))
            if ep is None:                                    # 休日発表は直後の営業日に丸める
                later = [d for d in all_dates if d >= pd.Timestamp(e)]
                ep = date_pos[later[0]] if later else None
            if ep is not None and abs(ep - here) <= k:
                return True
        return False

    for i, date in enumerate(all_dates):
        exited_today: set[str] = set()

        # ---------- 1) 前日に決めた注文を、当日の始値で執行する ----------
        for sym, pos in list(positions.items()):
            if pos.pending_exit is None or date not in data[sym].index:
                continue
            fill = float(data[sym].at[date, "open"]) * (1 - cfg.slippage_pct / 100)
            t = pos.trade
            t.exit_date, t.exit_price, t.exit_reason = date, fill, pos.pending_exit
            cash += fill * t.shares * (1 - cfg.commission_pct / 100)
            trades.append(t)
            del positions[sym]
            exited_today.add(sym)

        for pe in pending:
            sym, df = pe.symbol, data[pe.symbol]
            if date not in df.index or sym in positions:
                continue
            if sym in exited_today:                       # RM-006: 同一銘柄の同日再建てはしない
                reject("差金決済規制（同一銘柄の同日再建て）")
                continue
            fill = float(df.at[date, "open"]) * (1 + cfg.slippage_pct / 100)
            stop_price = pe.stop_price
            if "close_at_entry" in spec.stop_initial:
                stop_price = _stop_at_fill(spec.id, spec.stop_initial, precomputed[sym]["ns"], pe.decision_date, fill)
            if pd.isna(stop_price) or fill <= stop_price:
                reject("寄付が損切り価格を下回った")
                continue
            # 約定価格でサイズを再計算する。ギャップで実リスクが予算を超えないようにする（RM-002）
            risk_per_share = (fill - stop_price) * cfg.assumed_loss_multiple
            shares = min(pe.shares,
                         int(math.floor(pe.risk_budget / risk_per_share)),
                         int(math.floor(pe.value_cap / fill)))
            if shares < 1:
                reject("約定価格ではサイズが1株未満")
                continue
            cost = fill * shares * (1 + cfg.commission_pct / 100)
            if cost > cash:
                shares = int(math.floor(cash / (fill * (1 + cfg.commission_pct / 100))))
                if shares < 1:
                    reject("買付余力不足")
                    continue
                cost = fill * shares * (1 + cfg.commission_pct / 100)
            cash -= cost
            positions[sym] = Position(
                trade=Trade(sym, spec.id, date, fill, shares, stop_price),
                stop_price=stop_price, last_close=fill,
            )
        pending = []

        # ---------- 2) 当日終値で評価し、翌営業日の注文を決める ----------
        for sym, pos in positions.items():
            if date in data[sym].index:
                pos.last_close = float(data[sym].at[date, "close"])
        equity = cash + sum(p.last_close * p.trade.shares for p in positions.values())
        equity_rows.append((date, equity))

        # データが途切れた銘柄は、その最終値で強制手仕舞いする（資産が消えないように）
        for sym, pos in list(positions.items()):
            if precomputed[sym]["last_date"] == date and i < len(all_dates) - 1:
                t = pos.trade
                t.exit_date, t.exit_price, t.exit_reason = date, pos.last_close, "データ終了"
                cash += pos.last_close * t.shares
                trades.append(t)
                del positions[sym]
                warnings.append(f"{sym}: {date.date()} でデータが終了したため最終値で手仕舞い")

        if i == len(all_dates) - 1:
            break

        # 2-a) 保有中の手仕舞い判定
        for sym, pos in positions.items():
            df, pre = data[sym], precomputed[sym]
            if date not in df.index:
                continue
            pos.bars_held += 1
            close = pos.last_close
            risk = pos.trade.risk_per_share
            r_now = (close - pos.trade.entry_price) / risk if risk > 0 else 0.0

            # トレーリングストップ。発動条件（+2R 等）があれば到達後のみ
            if pre["stop_trailing"] is not None:
                if trailing_at is None or pos.trailing_active or r_now >= trailing_at:
                    pos.trailing_active = True
                    new_stop = pre["stop_trailing"].get(date, np.nan)
                    if not pd.isna(new_stop):
                        pos.stop_price = (max(pos.stop_price, float(new_stop))
                                          if spec.stop_ratchet else float(new_stop))

            reason = None
            if close < pos.stop_price:
                reason = "論理ストップ"          # RM-001a: 翌執行枠で成行
            elif (cfg.earnings_dates and cfg.earnings_policy != "entry_only"
                  and in_earnings_blackout(sym, all_dates[i + 1])
                  and not (cfg.earnings_policy == "cushion" and r_now >= cfg.earnings_cushion_r)):
                reason = "決算跨ぎ回避"          # RM-001d（方針は cfg.earnings_policy）
            else:
                for rule in spec.exit_rules:
                    cond = rule["condition"]
                    only_rebalance = rebalance_days is not None and "リバランス" in str(rule.get("applies_at", ""))
                    if only_rebalance and date not in rebalance_days:
                        continue
                    if any(k in cond for k in _POSITION_SCOPED):
                        hit = _position_rule(cond, holding_days=pos.bars_held, unrealized_r=r_now)
                    else:
                        hit = bool(pre["exits"].get(rule["name"], pd.Series(dtype=bool)).get(date, False))
                    if hit:
                        reason = rule["name"]
                        break
            pos.pending_exit = reason

        # 2-b) 新規エントリーの判定
        day_pnl_pct = (equity - prev_equity) / prev_equity * 100 if prev_equity > 0 else 0.0
        prev_equity = equity
        peak_equity = max(peak_equity, equity)
        dd_pct = (1 - equity / peak_equity) * 100 if peak_equity > 0 else 0.0
        size_mult = 1.0
        for threshold, mult in cfg.drawdown_derisk:            # RM-022: 深いDDほど小さく
            if dd_pct >= threshold:
                size_mult = mult
                break
        if rebalance_days is not None and date not in rebalance_days:
            continue                                        # 月次リバランス: 月末以外は建てない
        if cfg.daily_loss_limit_pct is not None and day_pnl_pct <= -float(cfg.daily_loss_limit_pct):
            reject("日次損失上限に到達（当日の新規建て停止）")   # RM-020
            continue
        open_after_exits = sum(1 for p in positions.values() if p.pending_exit is None)
        # 損切りが建値の上まで切り上がった玉はリスク0として数える（負のヒートにしない）
        heat = sum(max(0.0, p.trade.entry_price - p.stop_price) * p.trade.shares * cfg.assumed_loss_multiple
                   for p in positions.values() if p.pending_exit is None)

        candidates: list[tuple[str, float, float, float]] = []   # (銘柄, 終値, 損切り, 優先度)
        for sym, df in data.items():
            if sym in positions or date not in df.index:
                continue
            pre = precomputed[sym]
            if not bool(pre["entry"].get(date, False)):
                continue
            if in_earnings_blackout(sym, date) or in_earnings_blackout(sym, all_dates[i + 1]):
                reject("決算発表前後のため見送り")
                continue
            stop = pre["stop_initial"]
            stop_price = float(stop.get(date, np.nan)) if isinstance(stop, pd.Series) else float(stop)
            close = float(df.at[date, "close"])
            if pd.isna(stop_price) or stop_price >= close:
                reject("損切り価格が不正")
                continue
            score = 0.0
            if precomputed[sym]["rank"] is not None:
                v = precomputed[sym]["rank"].get(date, np.nan)
                score = float(v) if not pd.isna(v) else float("-inf")
            candidates.append((sym, close, stop_price, score))

        # entry.rank に従って並べ、上限に達したら弱い候補から落とす（銘柄コード順にしない）
        if spec.rank is not None:
            candidates.sort(key=lambda c: c[3], reverse=spec.rank[1])
        for sym, close, stop_price, _score in candidates:
            if open_after_exits + len(pending) >= max_positions:
                reject("同時保有数の上限")
                continue
            risk_per_share = (close - stop_price) * cfg.assumed_loss_multiple
            risk_pct = spec.risk_pct if cfg.risk_pct_override is None else cfg.risk_pct_override
            budget = equity * risk_pct / 100 * size_mult
            value_cap = equity * cfg.max_position_pct / 100 * size_mult
            shares = min(int(math.floor(budget / risk_per_share)), int(math.floor(value_cap / close)))
            if shares < 1:
                reject("サイズが1株未満（リスク予算に対して株価が高すぎる）")
                continue
            if heat + risk_per_share * shares > equity * cfg.max_portfolio_heat_pct / 100:
                reject("ポートフォリオ・ヒートの上限")
                continue
            heat += risk_per_share * shares
            pending.append(_PendingEntry(sym, stop_price, shares, date, budget, value_cap))

    # 期末に残った建玉は最終終値で評価して閉じる
    last = all_dates[-1]
    for sym, pos in positions.items():
        t = pos.trade
        t.exit_date, t.exit_price, t.exit_reason = last, pos.last_close, "期末評価"
        trades.append(t)

    curve = pd.Series(dict(equity_rows)).sort_index()
    return BacktestResult(trades=trades, equity_curve=curve, rejections=rejections, warnings=warnings)
