"""戦略間の相関を測る（VR-045 / docs/18 §18.4 の事前予想との照合）。

測るもの:
  1. 資金曲線の日次リターンの相関（実際に同時に沈むか）
  2. 市場ベータを除いた残差リターンの相関（**VR-045 の判定はこちら**、docs/26 DEC-066）
     現物買いだけの戦略は、どれも「市場が上がれば増え、下がれば減る」共通部分（ベータ）を持つ。
     生の相関はその共通部分で底上げされ、シグナルの重なりが見えない。
     各戦略の日次リターンを指数リターンで回帰し、残差どうしの相関を取る。
  3. エントリー日×銘柄の重なり（Jaccard）（同じ日に同じ銘柄を買っているか）

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

MEASURE_RISK_PCT = 0.25     # 相関測定用の固定リスク率（%）。成績の DD はこの率での値


@dataclass
class CorrelationReport:
    strategies: list[str]
    return_corr: pd.DataFrame
    entry_jaccard: pd.DataFrame
    predicted: dict[tuple[str, str], float]
    metrics: dict[str, dict] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    residual_corr: pd.DataFrame | None = None      # 市場ベータを除いた残差の相関
    betas: dict[str, float] = field(default_factory=dict)
    earnings_symbols_matched: int | None = None

    def judged(self) -> pd.DataFrame:
        """VR-045 の判定に使う行列。残差相関が測れていればそれ、無ければ生の相関。"""
        return self.residual_corr if self.residual_corr is not None else self.return_corr

    def flags(self, threshold: float = 0.7, *, raw: bool = False) -> list[tuple[str, str, float]]:
        """同一因子として扱うべきペア（VR-045）。既定は残差相関で判定する（DEC-066）。"""
        mat = self.return_corr if raw else self.judged()
        out = []
        for a, b in itertools.combinations(self.strategies, 2):
            v = float(mat.loc[a, b])
            if not np.isnan(v) and v >= threshold:
                out.append((a, b, v))
        return out

    def to_dict(self) -> dict:
        pairs = []
        for a, b in itertools.combinations(self.strategies, 2):
            pairs.append({
                "pair": f"{a} x {b}",
                "return_corr": _r(self.return_corr.loc[a, b]),
                "residual_corr": _r(self.residual_corr.loc[a, b]) if self.residual_corr is not None else None,
                "entry_jaccard": _r(self.entry_jaccard.loc[a, b]),
                "predicted": self.predicted.get((a, b), self.predicted.get((b, a))),
            })
        return {"strategies": self.strategies, "pairs": pairs,
                "judged_on": "residual" if self.residual_corr is not None else "raw",
                "flags_over_0.7": [f"{a} x {b} = {v:.2f}" for a, b, v in self.flags()],
                "raw_flags_over_0.7": [f"{a} x {b} = {v:.2f}" for a, b, v in self.flags(raw=True)],
                "betas": {k: _r(v) for k, v in self.betas.items()},
                "measure_risk_pct": MEASURE_RISK_PCT,
                "earnings_symbols_matched": self.earnings_symbols_matched,
                "metrics": self.metrics, "skipped": self.skipped}


def market_residuals(curves: dict[str, pd.Series], index: pd.DataFrame | None
                     ) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """各戦略の日次リターンから市場ベータ×指数リターンを引いた残差と、ベータを返す。"""
    if index is None or "close" not in index.columns:
        return {}, {}
    mkt = index["close"].astype(float).pct_change()
    residuals: dict[str, pd.Series] = {}
    betas: dict[str, float] = {}
    for sid, r in curves.items():
        m = mkt.reindex(r.index).fillna(0.0)
        var = float(m.var())
        beta = float(r.cov(m) / var) if var > 0 else 0.0
        residuals[sid] = r - beta * m
        betas[sid] = beta
    return residuals, betas


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
            # 成績ではなく「どの日にどの銘柄を買うか」の構造を見るため、
            # 資金・同時保有数・日次損失上限・DD縮小といった資金側の制約を外す。
            # 戦略仕様に固有の上限（ST-09 の max_positions=5）は戦略の一部なので残す。
            # リスク率は仕様の値ではなく固定の小さい値（0.25%）にする。仕様の 2% だと1玉が資産の
            # 2割前後になり、数玉で買付余力が尽きて「取れるはずのシグナル」が落ちる（docs/27 §27.3）。
            # 相関は規模に依らないが、余力切れによる取捨選択には依存する。
            res = run(spec, data, index=index,
                      config=BacktestConfig.from_common(spec.common, initial_equity=equity,
                                                        slippage_pct=0.0, earnings_dates=earnings,
                                                        risk_pct_override=MEASURE_RISK_PCT,
                                                        max_position_pct=100.0, max_portfolio_heat_pct=100.0,
                                                        max_positions=10_000, daily_loss_limit_pct=None,
                                                        drawdown_derisk=[]))
        except UnsupportedSpec as e:
            skipped[spec.id] = str(e)
            continue
        curves[spec.id] = res.equity_curve.pct_change().fillna(0.0)
        entries[spec.id] = {(t.symbol, t.entry_date) for t in res.trades}
        metrics[spec.id] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.metrics().items()}

    ids = list(curves)
    ret = pd.DataFrame(curves).corr() if len(ids) >= 2 else pd.DataFrame(index=ids, columns=ids, dtype=float)
    residuals, betas = market_residuals(curves, index)
    res_corr = (pd.DataFrame(residuals).corr() if len(residuals) >= 2 else None)
    jac = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for a, b in itertools.combinations(ids, 2):
        u = entries[a] | entries[b]
        jac.loc[a, b] = jac.loc[b, a] = (len(entries[a] & entries[b]) / len(u)) if u else 0.0

    predicted: dict[tuple[str, str], float] = {}
    for spec in specs:
        for other, info in (spec.raw.get("expected_correlation") or {}).items():
            if isinstance(info, dict) and "predicted" in info:
                predicted[(spec.id, other)] = float(info["predicted"])
    return CorrelationReport(ids, ret, jac, predicted, metrics, skipped, res_corr, betas)


def format_table(rep: CorrelationReport) -> str:
    lines = ["  ペア                 生の相関  残差相関  予想   差    エントリー重なり"]
    for a, b in itertools.combinations(rep.strategies, 2):
        v = rep.return_corr.loc[a, b]
        rv = rep.residual_corr.loc[a, b] if rep.residual_corr is not None else np.nan
        judged = rv if rep.residual_corr is not None else v
        p = rep.predicted.get((a, b), rep.predicted.get((b, a)))
        j = rep.entry_jaccard.loc[a, b]
        diff = "" if p is None or np.isnan(judged) else f"{judged - p:+.2f}"
        mark = "  ← 0.7超（同一因子扱い）" if not np.isnan(judged) and judged >= 0.7 else ""
        rv_text = "   -" if np.isnan(rv) else f"{rv:>8.2f}"
        lines.append(f"  {a} x {b:<8} {v:>8.2f} {rv_text}  {('' if p is None else f'{p:.2f}'):>5} {diff:>6}  {j:>6.2f}{mark}")
    if rep.betas:
        lines.append("  市場ベータ: " + ", ".join(f"{k} {v:.2f}" for k, v in rep.betas.items()))
        lines.append("  （判定は残差相関。生の相関は市場ベータで底上げされる。docs/26 DEC-066）")
    for sid, why in rep.skipped.items():
        lines.append(f"  （{sid} は未実装のため除外: {why[:60]}）")
    return "\n".join(lines)


def save(rep: CorrelationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
