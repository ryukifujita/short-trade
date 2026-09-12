"""学習／検証の分割つきパラメータ探索（VR-020 / VR-021、docs/08）。

やること（docs/30 §30.2）:
  1. 各戦略が宣言した params_to_optimize の **範囲の両端と既定値**（各 3 水準）だけを格子にする。
     細かい刻みで探せば探すほど、たまたま当たった組合せが見つかる（過剰最適化）。自由度は宣言どおりに保つ。
  2. 学習期間（train_end まで）で格子を走らせ、**資金制約をほぼ外した構造測定**（correlate と同じ）で
     平均R が最大の組合せを選ぶ。取引数が min_trades 未満の組合せは選ばない。
  3. 検証期間（train_end の翌日から）で、**既定値と選んだ値の両方**を 5万円の運用条件で走らせる。
     選んだ値が既定値に勝ち、かつ G1 の基準を満たすかを見る。検証期間の結果は一度しか見ない。
  4. 格子全体の学習成績も保存する。プラトー（隣接する組合せも同じように良い）でなければ、
     最良点はノイズと見なす（G1 の「パラメータ感度がプラトー」）。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, UnsupportedSpec, run
from .correlate import MEASURE_RISK_PCT
from .spec import StrategySpec

G1 = {"取引数": 100, "プロフィットファクタ": 1.3, "最大DD": -15.0}


def grid_levels(param: dict[str, Any]) -> list[Any]:
    """宣言した範囲の両端と既定値（3 水準）。既定値が端なら中点を足して 3 つにする。"""
    lo, hi = param["range"]
    d = param["default"]
    vals = {lo, hi, d}
    if len(vals) < 3:
        mid = (lo + hi) / 2
        step = param.get("step")
        if step:
            mid = round(round(mid / step) * step, 10)
        vals.add(mid)
    out = sorted(vals)
    return [int(v) if isinstance(d, int) and float(v).is_integer() else v for v in out]


def _slice(data: dict[str, pd.DataFrame], start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in data.items():
        d = df
        if start is not None:
            d = d.loc[start:]
        if end is not None:
            d = d.loc[:end]
        if len(d) > 250:
            out[sym] = d
    return out


def _r(m: dict) -> dict:
    """丸める。無限大（負けが 0 件の PF）は JSON にできないので None にする。"""
    out = {}
    for k, v in m.items():
        if isinstance(v, float):
            out[k] = None if (v != v or v in (float("inf"), float("-inf"))) else round(v, 3)
        else:
            out[k] = v
    return out


@dataclass
class WalkForwardResult:
    strategy: str
    train_end: str
    grid: list[dict] = field(default_factory=list)
    best: dict | None = None
    defaults: dict | None = None
    test: dict = field(default_factory=dict)
    plateau_share: float | None = None
    g1: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "train_end": self.train_end, "defaults": self.defaults,
                "best": self.best, "plateau_share": self.plateau_share, "grid": self.grid,
                "test": self.test, "g1": self.g1, "note": self.note}


def g1_check(m: dict) -> dict:
    """G1 の数値基準（docs/08）。プラトーは別途。"""
    return {
        "取引数 >= 100": bool(m.get("取引数", 0) >= G1["取引数"]),
        "PF >= 1.3": bool(m.get("プロフィットファクタ", 0) >= G1["プロフィットファクタ"]),
        "期待値 > 0": bool(m.get("期待値", 0) > 0),
        "最大DD <= 15%": bool(m.get("最大DD", -100) >= G1["最大DD"]),
    }


def walk_forward(spec: StrategySpec, data: dict[str, pd.DataFrame], *, index: pd.DataFrame | None,
                 train_end: str, earnings=None, membership=None, equity: float = 50_000.0,
                 slippage_pct: float = 0.1, min_trades: int = 30, warmup_bars: int = 300,
                 progress=None) -> WalkForwardResult:
    res = WalkForwardResult(spec.id, train_end, defaults=dict(spec.params))
    te = pd.Timestamp(train_end)
    train = _slice(data, end=te)
    train_index = index.loc[:te] if index is not None else None

    def structural(cfg_spec: StrategySpec, d: dict, idx, **over) -> dict | None:
        cfg = BacktestConfig.from_common(cfg_spec.common, initial_equity=10_000_000.0, slippage_pct=0.0,
                                         earnings_dates=earnings, universe_membership=membership,
                                         risk_pct_override=MEASURE_RISK_PCT, max_position_pct=100.0,
                                         max_portfolio_heat_pct=100.0, max_positions=10_000,
                                         daily_loss_limit_pct=None, drawdown_derisk=[], **over)
        try:
            return run(cfg_spec, d, index=idx, config=cfg).metrics()
        except UnsupportedSpec as e:
            res.note = str(e)
            return None

    # 1〜2. 学習期間の格子
    names = [p["name"] for p in spec.param_specs]
    levels = [grid_levels(p) for p in spec.param_specs]
    combos = list(itertools.product(*levels)) if names else [()]
    best_score, best_params = None, None
    for i, combo in enumerate(combos, 1):
        params = dict(zip(names, combo))
        m = structural(spec.with_params(**params), train, train_index)
        if m is None:
            return res
        row = {"params": params, **_r({k: m.get(k) for k in ("取引数", "プロフィットファクタ", "平均R", "期待値", "最大DD")})}
        res.grid.append(row)
        score = m.get("平均R", float("-inf")) if m.get("取引数", 0) >= min_trades else float("-inf")
        if best_score is None or score > best_score:
            best_score, best_params = score, params
        if progress:
            progress(f"  [{i}/{len(combos)}] {params} → 取引 {m.get('取引数', 0)} PF {m.get('プロフィットファクタ', 0):.2f} 平均R {m.get('平均R', 0):+.3f}")
    ok = [g for g in res.grid if (g.get("プロフィットファクタ") or 0) >= 1.0 and (g.get("取引数") or 0) >= min_trades]
    res.plateau_share = round(len(ok) / len(res.grid), 3) if res.grid else None
    if best_score is None or best_score == float("-inf"):
        res.note = f"学習期間で取引数 {min_trades} 以上の組合せがありません"
        best_params = dict(spec.params)
    elif best_score <= 0:
        # 学習期間で勝てる組合せが無いのに「最もましな負け方」を選ぶのは選択ではない。既定値のままにする
        res.note = f"学習期間で平均R が正の組合せがありません（最良 {best_score:+.3f}）。既定値のままにします"
        best_params = dict(spec.params)
    res.best = best_params

    # 3. 検証期間（助走つき。train_end より前は建てない）
    dates = sorted({d for df in data.values() for d in df.index})
    pos = max(0, next((i for i, d in enumerate(dates) if d > te), len(dates)) - warmup_bars)
    test = _slice(data, start=dates[pos])
    test_index = index.loc[dates[pos]:] if index is not None else None
    for label, params in (("default", dict(spec.params)), ("best", best_params)):
        sp = spec.with_params(**params)
        cfg = BacktestConfig.from_common(sp.common, initial_equity=equity, slippage_pct=slippage_pct,
                                         earnings_dates=earnings, universe_membership=membership,
                                         entries_from=te + pd.Timedelta(days=1))
        try:
            m = run(sp, test, index=test_index, config=cfg).metrics()
        except UnsupportedSpec as e:
            res.note = str(e)
            return res
        ms = structural(sp, test, test_index, entries_from=te + pd.Timedelta(days=1))
        res.test[label] = {"params": params, "operating": _r(m), "structural": _r(ms or {})}
    res.g1 = {label: g1_check(v["operating"]) for label, v in res.test.items()}
    return res


def format_result(r: WalkForwardResult) -> str:
    lines = [f"  {r.strategy}: 学習 〜{r.train_end} / 検証 {r.train_end} 以降  既定 {r.defaults} → 選択 {r.best}"
             f"  プラトー率 {r.plateau_share}"]
    for label, v in r.test.items():
        m = v["operating"]
        lines.append(f"    検証[{label:<7}] 取引 {m.get('取引数', 0):>4} PF {m.get('プロフィットファクタ', 0):>5.2f} "
                     f"平均R {m.get('平均R', 0):>+6.3f} 期待値 {m.get('期待値', 0):>+8.1f} 最大DD {m.get('最大DD', 0):>6.1f}%"
                     f"  G1: {'合格' if all(r.g1.get(label, {}).values()) else '不合格'}")
    if r.note:
        lines.append(f"    注: {r.note}")
    return "\n".join(lines)


def save(results: list[WalkForwardResult], path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**(extra or {}), "results": [r.to_dict() for r in results]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
