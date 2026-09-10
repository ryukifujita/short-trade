"""イベント駆動バックテスタ。

設計の核（docs/18 §18.1）:
  - すべての戦略は「当日終値で判定 → 翌営業日9:00の成行」で執行する。
    判定日 t で参照できるのは t までの確定足だけであり、約定価格は t+1 の始値である。
  - S株は逆指値が置けないため、損切りは論理ストップ＋翌執行枠の成行（RM-001a）。
  - 想定損失は損切り幅の assumed_loss_multiple 倍で見積もる（RM-001b）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .indicators import build_namespace, evaluate
from .spec import StrategySpec

# ポジション単位でしか評価できない名前（式ではなく Python 側で判定する）
_POSITION_SCOPED = ("holding_days", "unrealized_r")


_XRANK_RE = re.compile(r"PCTRANK\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*universe\s*\)")


def _xrank_name(inner: str) -> str:
    return f"__XRANK_{inner}"


def _rewrite_xrank(expr: str) -> str:
    """`PCTRANK(x, universe)` を事前計算済みの名前に置き換える。"""
    return _XRANK_RE.sub(lambda m: _xrank_name(m.group(1)), expr)


def _xrank_targets(spec: StrategySpec) -> list[str]:
    """クロスセクショナル順位が必要な中間量の名前を集める。"""
    texts = list(spec.entry_conditions) + list(spec.regime_conditions)
    texts += [r["condition"] for r in spec.exit_rules]
    texts += [spec.stop_initial] + ([spec.stop_trailing] if spec.stop_trailing else [])
    names: list[str] = []
    for t in texts:
        names += _XRANK_RE.findall(t)
    return list(dict.fromkeys(names))


def _safe_eval(expr: str, namespace: dict) -> pd.Series | None:
    """評価できない式（未実装の検出器を参照するなど）は None を返す。

    None のまま損切り価格が決まらない戦略はエントリーできない。これは意図した挙動で、
    「損切り価格のないポジションを持たない」（DEC-004 / RM-001）を実装レベルで守るためである。
    """
    try:
        return evaluate(expr, namespace)
    except Exception:
        return None


def _stop_at_fill(expr: str, namespace: dict, decision_date, fill_price: float) -> float:
    """約定価格が確定したあとに損切り価格を確定させる。"""
    ns = dict(namespace, close_at_entry=fill_price)
    series = _safe_eval(expr, ns)
    if series is None:
        return float("nan")
    if isinstance(series, pd.Series):
        return float(series.get(decision_date, np.nan))
    return float(series)


def _align(series: pd.Series, target: pd.Index) -> pd.Series:
    """条件式の評価結果を銘柄の日付に整列する。

    指数（INDEX）を参照する条件は指数側の日付で返るため、銘柄の営業日へ寄せる。
    ffill を使うのは「直近に確定している判定を引き継ぐ」ためで、未来は参照しない。
    """
    if series.index.equals(target):
        return series.fillna(False).astype(bool)
    return series.reindex(series.index.union(target)).ffill().reindex(target).fillna(False).astype(bool)


class UnsupportedSpec(RuntimeError):
    """仕様が参照している要素がまだ実装されていない。

    黙って無視すると「シグナルが出ない戦略」に見えてしまい、検証結果を誤読する。
    未実装は未実装として、名前を挙げて落とす。
    """


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
    def holding_days(self) -> int:
        if self.exit_date is None:
            return 0
        return int((self.exit_date - self.entry_date).days)


@dataclass
class Position:
    trade: Trade
    stop_price: float
    bars_held: int = 0
    pending_exit: str | None = None    # 翌営業日の始値で手仕舞う理由


@dataclass
class BacktestConfig:
    initial_equity: float = 50_000.0
    slippage_pct: float = 0.10          # 片道。成行のため保守的に置く（VR-011）
    commission_pct: float = 0.0         # SBI証券ゼロ革命（PL-002 / CR-041）
    max_positions: int = 8
    max_position_pct: float = 15.0
    max_portfolio_heat_pct: float = 6.0
    assumed_loss_multiple: float = 2.0
    allow_fractional_shares: bool = True   # S株は1株単位

    @classmethod
    def from_common(cls, common: dict[str, Any], **overrides: Any) -> "BacktestConfig":
        risk = common.get("risk", {})
        cfg = cls(
            max_positions=int(risk.get("max_concurrent_positions", 8)),
            max_position_pct=float(risk.get("max_position_pct", 15)),
            max_portfolio_heat_pct=float(risk.get("max_portfolio_heat_pct", 6.0)),
            assumed_loss_multiple=float(risk.get("assumed_loss_multiple", 2.0)),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    rejections: dict[str, int] = field(default_factory=dict)

    # ---------- 指標（docs/12 の KPI 定義に合わせる） ----------
    def metrics(self) -> dict[str, float]:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return {"取引数": 0}
        pnl = np.array([t.pnl for t in closed])
        r = np.array([t.r_multiple for t in closed])
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_loss = -losses.sum()
        eq = self.equity_curve
        dd = (eq / eq.cummax() - 1.0).min() if len(eq) else 0.0
        return {
            "取引数": len(closed),
            "勝率": float(len(wins) / len(closed) * 100),
            "総損益": float(pnl.sum()),
            "平均利益": float(wins.mean()) if len(wins) else 0.0,
            "平均損失": float(losses.mean()) if len(losses) else 0.0,
            "プロフィットファクタ": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
            "平均R": float(r.mean()),
            "期待値": float(pnl.mean()),
            "最大DD": float(dd * 100),
            "最大連敗": _max_streak(pnl <= 0),
        }


def _max_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


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

    # --- 事前計算: 銘柄ごとに条件式を評価しておく（実運用でも同じ式を使う） ---
    if spec.unsupported_definitions:
        raise UnsupportedSpec(
            f"{spec.id}: 専用の検出器が必要な定義があります: "
            f"{', '.join(spec.unsupported_definitions)}"
        )

    # --- 事前パス: クロスセクショナル順位（銘柄間の相対順位）を計算する ---
    xrank_names = _xrank_targets(spec)
    xranks: dict[str, pd.DataFrame] = {}
    if xrank_names:
        raw_values: dict[str, dict[str, pd.Series]] = {n: {} for n in xrank_names}
        for sym, df in data.items():
            ns0 = build_namespace(df, index=index)
            for name, expr in spec.definitions.items():
                if isinstance(expr, str):
                    try:
                        ns0[name] = evaluate(expr, ns0)
                    except (NameError, SyntaxError):
                        pass
                else:
                    ns0[name] = expr
            for n in xrank_names:
                if n in ns0 and isinstance(ns0[n], pd.Series):
                    raw_values[n][sym] = ns0[n]
        for n in xrank_names:
            if not raw_values[n]:
                raise UnsupportedSpec(
                    f"{spec.id}: クロスセクショナル順位の対象 {n} を計算できません"
                )
            frame = pd.DataFrame(raw_values[n])
            # 各日について、その日の全銘柄のなかでのパーセンタイル（0〜100）
            xranks[n] = frame.rank(axis=1, pct=True) * 100.0

    precomputed: dict[str, dict[str, pd.Series]] = {}
    for sym, df in data.items():
        ns = build_namespace(df, index=index)
        for n, frame in xranks.items():
            if sym in frame.columns:
                ns[_xrank_name(n)] = frame[sym].reindex(df.index)
        # `definitions:` を先に評価して名前空間へ入れる（後の定義は前の定義を使える）
        for name, expr in spec.definitions.items():
            if not isinstance(expr, str):
                ns[name] = expr                      # 定数（件数・閾値など）
                continue
            try:
                ns[name] = evaluate(expr, ns)
            except NameError as e:
                raise UnsupportedSpec(
                    f"{spec.id}: 定義 {name} が未実装の要素を参照しています: {e}"
                ) from e
            except SyntaxError as e:
                raise UnsupportedSpec(
                    f"{spec.id}: 定義 {name} が式になっていません（説明文のままです）: {expr[:40]}…"
                ) from e
        entry = None
        for cond in spec.entry_conditions + spec.regime_conditions:
            try:
                raw_series = evaluate(_rewrite_xrank(cond), ns)
            except NameError as e:
                raise UnsupportedSpec(
                    f"{spec.id}: 条件 `{cond}` が未実装の要素を参照しています: {e}"
                ) from e
            s = _align(raw_series, df.index)
            entry = s if entry is None else (entry & s)
        exits: dict[str, pd.Series] = {}
        for rule in spec.exit_rules:
            cond = rule["condition"]
            if any(k in cond for k in _POSITION_SCOPED):
                continue  # ポジション単位。ループ内で評価する
            exits[rule["name"]] = _align(evaluate(_rewrite_xrank(cond), ns), df.index)
        # `close_at_entry` を含む損切り式は、判定日には約定価格が未確定である。
        # 判定時は当日終値を見積りとして使い（サイズ計算と候補選別）、
        # 約定後に実際の約定価格で置き換える（_stop_at_fill）。
        est_ns = dict(ns, close_at_entry=df["close"])
        precomputed[sym] = {
            "entry": entry if entry is not None else pd.Series(False, index=df.index),
            "stop_initial": _safe_eval(_rewrite_xrank(spec.stop_initial), est_ns),
            "stop_trailing": _safe_eval(_rewrite_xrank(spec.stop_trailing), ns) if spec.stop_trailing else None,
            "exits": exits,
            "ns": ns,
        }

    all_dates = sorted({d for df in data.values() for d in df.index})
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    cash = cfg.initial_equity
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    rejections: dict[str, int] = {}
    pending_entries: list[tuple[str, float, int, Any]] = []  # (銘柄, 損切り価格, 株数, 判定日)

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1
        if on_reject:
            on_reject(reason)

    for i, date in enumerate(all_dates):
        # ---------- 1) 前日に決めた注文を、当日の始値で執行する ----------
        for sym, pos in list(positions.items()):
            if pos.pending_exit is None:
                continue
            df = data[sym]
            if date not in df.index:
                continue
            fill = df.at[date, "open"] * (1 - cfg.slippage_pct / 100)
            t = pos.trade
            t.exit_date, t.exit_price, t.exit_reason = date, fill, pos.pending_exit
            cash += fill * t.shares * (1 - cfg.commission_pct / 100)
            trades.append(t)
            del positions[sym]

        for sym, stop_price, shares, decision_date in pending_entries:
            df = data[sym]
            if date not in df.index or sym in positions:
                continue
            fill = df.at[date, "open"] * (1 + cfg.slippage_pct / 100)
            if "close_at_entry" in spec.stop_initial:
                stop_price = _stop_at_fill(
                    spec.stop_initial, precomputed[sym]["ns"], decision_date, fill
                )
                if pd.isna(stop_price) or stop_price >= fill:
                    reject("約定価格に対して損切り価格が不正")
                    continue
            cost = fill * shares * (1 + cfg.commission_pct / 100)
            if cost > cash:
                reject("買付余力不足")
                continue
            if fill <= stop_price:                       # 寄付でストップを割り込んだ
                reject("寄付が損切り価格を下回った")
                continue
            cash -= cost
            positions[sym] = Position(
                trade=Trade(sym, spec.id, date, fill, shares, stop_price),
                stop_price=stop_price,
            )
        pending_entries = []

        # ---------- 2) 当日終値で評価し、翌営業日の注文を決める ----------
        equity = cash + sum(
            data[s].at[date, "close"] * p.trade.shares
            for s, p in positions.items() if date in data[s].index
        )
        equity_rows.append((date, equity))
        if i == len(all_dates) - 1:
            break

        # 2-a) 保有中の手仕舞い判定
        for sym, pos in positions.items():
            df, pre = data[sym], precomputed[sym]
            if date not in df.index:
                continue
            pos.bars_held += 1
            close = df.at[date, "close"]

            # トレーリングストップ（ratchet=true なら切り上げのみ）
            if pre["stop_trailing"] is not None:
                new_stop = pre["stop_trailing"].get(date, np.nan)
                if not pd.isna(new_stop):
                    pos.stop_price = (
                        max(pos.stop_price, float(new_stop))
                        if spec.stop_ratchet else float(new_stop)
                    )

            reason = None
            if close < pos.stop_price:
                reason = "論理ストップ"          # RM-001a: 翌執行枠で成行
            else:
                for rule in spec.exit_rules:
                    cond = rule["condition"]
                    if "holding_days" in cond:
                        limit = int(cond.split(">=")[1])
                        if pos.bars_held >= limit:
                            if "unrealized_r" in cond:
                                risk = pos.trade.risk_per_share
                                r_now = (close - pos.trade.entry_price) / risk if risk > 0 else 0
                                if r_now < float(cond.split("unrealized_r <")[1].split()[0]):
                                    reason = rule["name"]
                            else:
                                reason = rule["name"]
                    elif pre["exits"].get(rule["name"], pd.Series(dtype=bool)).get(date, False):
                        reason = rule["name"]
                    if reason:
                        break
            pos.pending_exit = reason

        # 2-b) 新規エントリーの判定
        open_after_exits = sum(1 for p in positions.values() if p.pending_exit is None)
        heat = sum(
            (p.trade.entry_price - p.stop_price) * p.trade.shares * cfg.assumed_loss_multiple
            for p in positions.values() if p.pending_exit is None
        )
        candidates: list[tuple[str, float, float]] = []   # (銘柄, 終値, 損切り価格)
        for sym, df in data.items():
            if sym in positions or date not in df.index:
                continue
            pre = precomputed[sym]
            if not bool(pre["entry"].get(date, False)):
                continue
            stop = pre["stop_initial"]
            stop_price = float(stop.get(date, np.nan)) if stop is not None else np.nan
            close = float(df.at[date, "close"])
            if pd.isna(stop_price) or stop_price >= close:
                reject("損切り価格が不正")
                continue
            candidates.append((sym, close, stop_price))

        for sym, close, stop_price in candidates:
            if open_after_exits + len(pending_entries) >= cfg.max_positions:
                reject("同時保有数の上限")
                continue
            risk_per_share = (close - stop_price) * cfg.assumed_loss_multiple
            budget = equity * spec.risk_pct / 100
            shares = int(math.floor(budget / risk_per_share))
            cap = int(math.floor(equity * cfg.max_position_pct / 100 / close))
            shares = min(shares, cap)
            if shares < 1:
                reject("サイズが1株未満")
                continue
            if heat + risk_per_share * shares > equity * cfg.max_portfolio_heat_pct / 100:
                reject("ポートフォリオ・ヒートの上限")
                continue
            heat += risk_per_share * shares
            pending_entries.append((sym, stop_price, shares, date))

    # 期末に残った建玉は最終終値で評価して閉じる
    last = all_dates[-1]
    for sym, pos in positions.items():
        if last in data[sym].index:
            t = pos.trade
            t.exit_date, t.exit_price, t.exit_reason = last, float(data[sym].at[last, "close"]), "期末評価"
            trades.append(t)

    curve = pd.Series(dict(equity_rows)).sort_index()
    return BacktestResult(trades=trades, equity_curve=curve, rejections=rejections)
