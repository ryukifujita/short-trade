"""戦略間の相関を測る（VR-045 / docs/18 §18.4 の事前予想との照合）。

測るもの:
  1. 資金曲線の日次リターンの相関（実際に同時に沈むか）
  2. エントリー日×銘柄の重なり（Jaccard）（同じ日に同じ銘柄を買っているか）

注意: 5万円のサイズ制約下では「1株未満」の却下が支配的になり、シグナル構造が見えない。
相関は **サイズ制約をほぼ外した資金（既定 1,000万円）** で測る。目的は成績ではなく構造の把握。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, UnsupportedSpec, run
from .spec import StrategySpec


@dataclass
class CorrelationReport:
    strategies: list[str]
    return_corr: pd.DataFrame
    entry_jaccard: pd.DataFrame
    predicted: dict[tuple[str, str], float]
    metrics: dict[str, dict] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def flags(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """同一因子として扱うべきペア（VR-045）。"""
        out = []
        for a, b in itertools.combinations(self.strategies, 2):
            v = float(self.return_corr.loc[a, b])
            if not np.isnan(v) and v >= threshold:
                out.append((a, b, v))
        return out

    def to_dict(self) -> dict:
        pairs = []
        for a, b in itertools.combinations(self.strategies, 2):
            pairs.append({
                "pair": f"{a} x {b}",
                "return_corr": _r(self.return_corr.loc[a, b]),
                "entry_jaccard": _r(self.entry_jaccard.loc[a, b]),
                "predicted": self.predicted.get((a, b), self.predicted.get((b, a))),
            })
        return {"strategies": self.strategies, "pairs": pairs,
                "flags_over_0.7": [f"{a} x {b} = {v:.2f}" for a, b, v in self.flags()],
                "metrics": self.metrics, "skipped": self.skipped}


def _r(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)


def measure(specs: list[StrategySpec], data: dict[str, pd.DataFrame], *,
            index: pd.DataFrame | None, equity: float = 10_000_000.0,
            earnings=None) -> CorrelationReport:
    curves: dict[str, pd.Series] = {}
    entries: dict[str, set[tuple[str, pd.Timestamp]]] = {}
    metrics: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for spec in specs:
        try:
            res = run(spec, data, index=index,
                      config=BacktestConfig.from_common(spec.common, initial_equity=equity,
                                                        slippage_pct=0.0, earnings_dates=earnings,
                                                        max_position_pct=100.0, max_portfolio_heat_pct=100.0))
        except UnsupportedSpec as e:
            skipped[spec.id] = str(e)
            continue
        curves[spec.id] = res.equity_curve.pct_change().fillna(0.0)
        entries[spec.id] = {(t.symbol, t.entry_date) for t in res.trades}
        metrics[spec.id] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.metrics().items()}

    ids = list(curves)
    ret = pd.DataFrame(curves).corr() if len(ids) >= 2 else pd.DataFrame(index=ids, columns=ids, dtype=float)
    jac = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for a, b in itertools.combinations(ids, 2):
        u = entries[a] | entries[b]
        jac.loc[a, b] = jac.loc[b, a] = (len(entries[a] & entries[b]) / len(u)) if u else 0.0

    predicted: dict[tuple[str, str], float] = {}
    for spec in specs:
        for other, info in (spec.raw.get("expected_correlation") or {}).items():
            if isinstance(info, dict) and "predicted" in info:
                predicted[(spec.id, other)] = float(info["predicted"])
    return CorrelationReport(ids, ret, jac, predicted, metrics, skipped)


def format_table(rep: CorrelationReport) -> str:
    lines = ["  ペア                 実測(日次)  予想   差    エントリー重なり"]
    for a, b in itertools.combinations(rep.strategies, 2):
        v = rep.return_corr.loc[a, b]
        p = rep.predicted.get((a, b), rep.predicted.get((b, a)))
        j = rep.entry_jaccard.loc[a, b]
        diff = "" if p is None or np.isnan(v) else f"{v - p:+.2f}"
        mark = "  ← 0.7超（同一因子扱い）" if not np.isnan(v) and v >= 0.7 else ""
        lines.append(f"  {a} x {b:<8} {v:>8.2f}  {('' if p is None else f'{p:.2f}'):>5} {diff:>6}  {j:>6.2f}{mark}")
    for sid, why in rep.skipped.items():
        lines.append(f"  （{sid} は未実装のため除外: {why[:60]}）")
    return "\n".join(lines)


def save(rep: CorrelationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
