"""条件ファネル: 戦略の各条件が「何銘柄・何日」通るかを数える。

目的: 「取引が2件しかない」ときに、どの条件で候補が消えているかを見る。
      成績ではなく、条件の厳しさの内訳を出す診断であり、バックテストの判定ロジックそのものを使う。

出力の読み方:
  - pass_days   … その条件 **単独** で真になった銘柄×日の数
  - cumulative  … その条件 **まで** をすべて AND にしたときの残り
  cumulative が大きく減る行が、候補を絞っている条件である。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import (BacktestConfig, _align, _eval_or_unsupported, _rewrite_xrank,
                       cross_sectional_ranks, regime_flag, symbol_namespace, universe_filters,
                       validate_bars)
from .spec import StrategySpec


@dataclass
class FunnelStep:
    kind: str            # "universe" | "entry" | "regime"
    text: str
    pass_days: int
    pass_symbols: int
    cumulative_days: int
    cumulative_symbols: int


@dataclass
class Funnel:
    strategy: str
    name: str
    symbols: int
    symbol_days: int
    steps: list[FunnelStep] = field(default_factory=list)

    @property
    def final_days(self) -> int:
        return self.steps[-1].cumulative_days if self.steps else self.symbol_days

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "name": self.name, "symbols": self.symbols,
            "symbol_days": self.symbol_days, "final_signal_days": self.final_days,
            "steps": [vars(s) for s in self.steps],
        }


def funnel(spec: StrategySpec, data: dict[str, pd.DataFrame], *,
           index: pd.DataFrame | None = None, config: BacktestConfig | None = None) -> Funnel:
    cfg = config or BacktestConfig.from_common(spec.common)
    data = {sym: validate_bars(sym, df) for sym, df in data.items()}
    xranks = cross_sectional_ranks(spec, data, index)

    # 各ステップの旗を銘柄ごとに集める。ステップの並びは run() と同じ。
    labels: list[tuple[str, str]] = []
    per_symbol: dict[str, list[pd.Series]] = {}
    for sym, df in data.items():
        ns = symbol_namespace(spec, sym, df, index, xranks)
        flags: list[pd.Series] = []
        first = not labels
        for text, flag in universe_filters(cfg, df, sym):
            if first:
                labels.append(("universe", text))
            flags.append(flag.astype(bool))
        for cond in spec.entry_conditions:
            if first:
                labels.append(("entry", cond))
            flags.append(_align(_eval_or_unsupported(spec.id, "条件", _rewrite_xrank(cond), ns), df.index).astype(bool))
        for cond in spec.regime_conditions:
            if first:
                labels.append(("regime", cond))
            flags.append(regime_flag(spec, cfg, cond, ns, df.index).astype(bool))
        per_symbol[sym] = flags

    total_days = sum(len(df) for df in data.values())
    out = Funnel(spec.id, spec.name, len(data), total_days)
    for i, (kind, text) in enumerate(labels):
        pass_days = pass_syms = cum_days = cum_syms = 0
        for sym, flags in per_symbol.items():
            single = flags[i]
            cum = flags[0].copy()
            for f in flags[1:i + 1]:
                cum &= f
            p, c = int(single.sum()), int(cum.sum())
            pass_days += p
            cum_days += c
            pass_syms += p > 0
            cum_syms += c > 0
        out.steps.append(FunnelStep(kind, text, pass_days, pass_syms, cum_days, cum_syms))
    return out


def format_funnel(f: Funnel) -> str:
    lines = [f"  {f.strategy} {f.name}: {f.symbols} 銘柄 × 延べ {f.symbol_days:,} 日 → 最終シグナル {f.final_days:,} 日"]
    lines.append(f"  {'種別':<8} {'単独で真':>10} {'ここまでAND':>12} {'銘柄':>5}  条件")
    for s in f.steps:
        kind = {"universe": "ユニバース", "entry": "エントリー", "regime": "レジーム"}[s.kind]
        lines.append(f"  {kind:<8} {s.pass_days:>10,} {s.cumulative_days:>12,} {s.cumulative_symbols:>5}  {s.text}")
    return "\n".join(lines)
