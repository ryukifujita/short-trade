"""式では書けないパターン検出器。仕様の `definitions:` から `{detector: 名前, ...引数}` で呼ぶ。

原則:
  - 因果性: 日付 t の値は t までの確定足だけから決まる（ルックアヘッド禁止・VR-002）。
    ジグザグの転換点は「反転が閾値を超えて確認された日」に初めて見える。
  - 出力は df.index に整列した DataFrame。列は仕様側が `名前.列` で参照する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ATR

__all__ = ["vcp", "DETECTORS"]


def _causal_zigzag(high: pd.Series, low: pd.Series, threshold: pd.Series):
    """反転が threshold（絶対値）を超えた時点で確定する転換点を、確定日つきで返す。

    返り値: list of (kind, pivot_pos, pivot_price, confirmed_pos)
      kind = "H"（山）| "L"（谷）
    """
    h, l, th = high.to_numpy(), low.to_numpy(), threshold.to_numpy()
    n = len(h)
    pivots = []
    if n == 0:
        return pivots
    # 最初は方向未定。仮の極値を追いながら、閾値を超えた反転で確定する
    trend = 0            # +1: 上昇中（山を探す）、-1: 下降中（谷を探す）
    ext_pos, ext_price = 0, h[0]
    ext_pos_l, ext_price_l = 0, l[0]
    for i in range(1, n):
        t = th[i] if not np.isnan(th[i]) else np.nan
        if np.isnan(t):
            # ATR が埋まる前は極値だけ更新する
            if h[i] > ext_price:
                ext_pos, ext_price = i, h[i]
            if l[i] < ext_price_l:
                ext_pos_l, ext_price_l = i, l[i]
            continue
        if trend >= 0:
            if h[i] > ext_price:
                ext_pos, ext_price = i, h[i]
            elif ext_price - l[i] > t:
                # 山が確定。以後は谷を探す
                pivots.append(("H", ext_pos, ext_price, i))
                trend = -1
                ext_pos_l, ext_price_l = i, l[i]
                continue
        if trend <= 0:
            if l[i] < ext_price_l:
                ext_pos_l, ext_price_l = i, l[i]
            elif h[i] - ext_price_l > t:
                pivots.append(("L", ext_pos_l, ext_price_l, i))
                trend = 1
                ext_pos, ext_price = i, h[i]
    return pivots


def vcp(df: pd.DataFrame, *, lookback: int = 90, atr_period: int = 20,
        atr_mult: float = 1.5) -> pd.DataFrame:
    """ボラティリティ収縮パターン（ST-04）。

    日付 t について、直近 lookback 本の中で **t までに確定した** 山→谷の押しを新しい順に
    p3, p2, p1（%）として返す。出来高は各押しの区間平均。pivot は最新の山、p3_low は最新の谷。
    押しが 3 つ未満なら contractions がその数になり、条件側で弾かれる。
    """
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    thr = ATR(atr_period, high=high, low=low, close=close) * atr_mult
    pivots = _causal_zigzag(high, low, thr)
    n = len(df)
    vol = volume.to_numpy()

    out = {k: np.full(n, np.nan) for k in ("p1", "p2", "p3", "v1", "v2", "v3", "pivot", "p3_low")}
    contractions = np.zeros(n, dtype=int)

    # 確定済みの押し（山→直後の谷）を確定日順に並べる
    legs = []   # (confirmed_pos, peak_pos, peak_price, trough_pos, trough_price)
    for a, b in zip(pivots, pivots[1:]):
        if a[0] == "H" and b[0] == "L":
            legs.append((b[3], a[1], a[2], b[1], b[2]))

    j = 0
    visible: list[tuple] = []
    for i in range(n):
        while j < len(legs) and legs[j][0] <= i:
            visible.append(legs[j])
            j += 1
        # 窓内（山の位置が i-lookback 以降）の押しだけを対象にする
        window = [lg for lg in visible if lg[1] >= i - lookback]
        contractions[i] = len(window)
        if not window:
            continue
        recent = window[-3:]
        depths = [(pk - tr) / pk * 100.0 for (_, _, pk, _, tr) in recent]
        vols = [float(vol[pp:tp + 1].mean()) for (_, pp, _, tp, _) in recent]
        # 新しい順に p3, p2, p1 へ割り当てる（不足分は NaN のまま）
        names = ["p3", "p2", "p1"][: len(recent)]
        for name, d, v in zip(names, depths[::-1], vols[::-1]):
            out[name][i] = d
            out["v" + name[1]][i] = v
        out["pivot"][i] = recent[-1][2]
        out["p3_low"][i] = recent[-1][4]

    res = pd.DataFrame(out, index=df.index)
    res["contractions"] = contractions
    return res


DETECTORS = {"vcp": vcp}
