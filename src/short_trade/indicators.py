"""catalog/schema.md の記法を実装する指標エンジン。

重要な原則（VR-001 / FR-143 / FR-144）:
  - バックテストと実運用でこの実装を共有する。二重実装を禁止する。
  - すべての指標は「確定足のみ」を参照する。当日を含む窓は当日の確定値までしか使わない。
  - 未来を参照しないことは tests/test_no_lookahead.py で検査する。
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

__all__ = [
    "SMA", "EMA", "ATR", "RSI", "HIGHEST", "LOWEST", "PCTRANK", "SHIFT", "STREAK",
    "expand_shifts", "normalize_logical", "expand_chained_comparison", "evaluate",
]


def SMA(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def EMA(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def HIGHEST(x: pd.Series, n: int) -> pd.Series:
    """直近 n 本の最高値（当日を含む）。"""
    return x.rolling(n, min_periods=n).max()


def LOWEST(x: pd.Series, n: int) -> pd.Series:
    """直近 n 本の最安値（当日を含む）。"""
    return x.rolling(n, min_periods=n).min()


def SHIFT(x: pd.Series, n: int) -> pd.Series:
    """n 営業日前の値。schema.md の `expr[-n]` はこれに展開される。"""
    return x.shift(n)


def TR(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def ATR(n: int, *, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder の平滑化による ATR（ST-46 / J. Welles Wilder）。"""
    tr = TR(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def RSI(x: pd.Series, n: int) -> pd.Series:
    """Wilder の RSI。n=2 でも安定するよう平滑化に ewm を使う。"""
    delta = x.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # 下落が皆無の区間は RSI=100 とする
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def STREAK(cond: pd.Series) -> pd.Series:
    """条件が連続して成立している本数（当日を含む）。不成立で 0 に戻る。"""
    c = cond.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(c), dtype=float)
    run = 0
    for i, v in enumerate(c):
        run = run + 1 if v else 0
        out[i] = run
    return pd.Series(out, index=cond.index)


def PCTRANK(x: pd.Series, n: int) -> pd.Series:
    """直近 n 本の分布における当日値のパーセンタイル（0〜100）。当日を含む。"""
    return x.rolling(n, min_periods=n).rank(pct=True) * 100.0


# ---------------------------------------------------------------- 式の評価

_SHIFT_RE = re.compile(r"\[-(\d+)\]")


def _split_top_level(expr: str, sep: str) -> list[str]:
    """括弧の外側にある sep（" and " / " or "）で式を分割する。"""
    parts, depth, start, i = [], 0, 0, 0
    n, m = len(expr), len(sep)
    while i < n:
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and expr.startswith(sep, i):
            parts.append(expr[start:i])
            i += m
            start = i
            continue
        i += 1
    parts.append(expr[start:])
    return parts


_CMP_OPS = ("<=", ">=", "<", ">")


def _split_top_level_cmp(expr: str) -> list[str]:
    """括弧の外側にある比較演算子で式を分割し、[被演算子, 演算子, 被演算子, ...] を返す。"""
    parts, depth, start, i = [], 0, 0, 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            for op in _CMP_OPS:
                if expr.startswith(op, i) and not (op in ("<", ">") and expr[i + 1:i + 2] == "="):
                    parts.append(expr[start:i].strip())
                    parts.append(op)
                    i += len(op)
                    start = i
                    break
            else:
                i += 1
                continue
            continue
        i += 1
    parts.append(expr[start:].strip())
    return parts


def expand_chained_comparison(expr: str) -> str:
    """`a > b > c` を `(a > b) & (b > c)` に展開する。

    Python の連鎖比較は内部で `and` を使うため、Series には使えない。
    仕様には `vcp.p1 > vcp.p2 > vcp.p3` のように自然に書けるようにしておく。
    """
    parts = _split_top_level_cmp(expr)
    if len(parts) < 5:            # 比較が1つ以下なら連鎖ではない
        return expr
    operands, ops = parts[0::2], parts[1::2]
    return " & ".join(f"({a} {op} {b})" for a, op, b in zip(operands, ops, operands[1:]))


def normalize_logical(expr: str) -> str:
    """`and` / `or` を Series 用の `&` / `|` に変換する。

    pandas の Series は Python の `and` を評価できない。また `&` は比較より
    優先順位が高いため、各項を括弧で包んでから連結する必要がある。
    仕様ファイルは人が読み書きするので `and` のまま書けるようにしておく。
    """
    for sep, op in ((" or ", " | "), (" and ", " & ")):
        parts = _split_top_level(expr, sep)
        if len(parts) > 1:
            return op.join(f"({normalize_logical(p.strip())})" for p in parts)
    return expand_chained_comparison(expr)


def expand_shifts(expr: str) -> str:
    """`EXPR[-n]` を `SHIFT(EXPR, n)` に書き換える。

    `[-n]` の直前にある式の開始位置を、括弧の対応を数えながら後ろ向きに探す。
    例: "HIGHEST(high, 20)[-1]" -> "SHIFT(HIGHEST(high, 20), 1)"
        "close[-2]"             -> "SHIFT(close, 2)"
    """
    while True:
        m = _SHIFT_RE.search(expr)
        if m is None:
            return expr
        n = m.group(1)
        start = _expr_start(expr, m.start())
        expr = (
            expr[:start]
            + f"SHIFT({expr[start:m.start()]}, {n})"
            + expr[m.end():]
        )


def _expr_start(expr: str, end: int) -> int:
    """expr[:end] の末尾にある単一の式（識別子・属性参照・関数呼び出し）の開始位置。"""
    i = end - 1
    depth = 0
    while i >= 0:
        c = expr[i]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth == 0:
                return i + 1
            depth -= 1
        elif depth == 0 and not (c.isalnum() or c in "_."):
            return i + 1
        i -= 1
    return 0


class _Frame:
    """`INDEX.close` のような属性アクセスを可能にする薄いラッパ。"""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getattr__(self, name: str) -> pd.Series:
        try:
            return self._df[name]
        except KeyError as e:  # pragma: no cover - 設定ミス時のみ
            raise AttributeError(f"列 {name} がありません") from e


def _elementwise(fn):
    """`max(a, b)` を要素ごとに評価する。Python 組み込みの max は Series を比較できない。"""
    def apply(*args):
        out = args[0]
        for a in args[1:]:
            out = fn(out, a)
        return out
    return apply


_SAFE_BUILTINS = {
    "abs": abs, "True": True, "False": False,
    "min": _elementwise(np.minimum), "max": _elementwise(np.maximum),
}


def build_namespace(df: pd.DataFrame, *, index: pd.DataFrame | None = None,
                    extra: dict | None = None) -> dict:
    """式の評価に使う名前空間を組み立てる。

    df は日付昇順・重複なしの OHLCV。index は市場全体（TOPIX 等）で、df の日付に整列させる。
    """
    def _rsi(a, b=None):
        """`RSI(n)` は終値のRSI、`RSI(x, n)` は任意系列のRSI。schema.md の記法に合わせる。"""
        return RSI(df["close"], a) if b is None else RSI(a, b)

    ns: dict = {
        "SMA": SMA, "EMA": EMA, "RSI": _rsi,
        "HIGHEST": HIGHEST, "LOWEST": LOWEST, "PCTRANK": PCTRANK, "SHIFT": SHIFT, "STREAK": STREAK,
        "ATR": lambda n: ATR(n, high=df["high"], low=df["low"], close=df["close"]),
        "__builtins__": _SAFE_BUILTINS,
    }
    for col in df.columns:
        ns[col] = df[col]
    if index is not None:
        # 指数は元の全履歴のまま束縛する。df の日付範囲が短くても SMA(200) が計算できるようにするため。
        # 評価結果の日付整列は呼び出し側（backtest.align）で行う。
        ns["INDEX"] = _Frame(index)
    if extra:
        ns.update(extra)
    return ns


def evaluate(expr: str, namespace: dict) -> pd.Series:
    """条件式を評価して bool の Series を返す。

    式は catalog/rules/*.yaml に書かれたものだけを想定する（利用者入力ではない）。
    それでも builtins は最小限に制限する。
    """
    prepared = expand_shifts(normalize_logical(expr))
    result = eval(prepared, namespace)  # noqa: S307 - 自前の仕様ファイルのみ
    if isinstance(result, pd.Series):
        return result
    # スカラー（定数条件）は全期間に broadcast する
    any_series = next(v for v in namespace.values() if isinstance(v, pd.Series))
    return pd.Series(result, index=any_series.index)
