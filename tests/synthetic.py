"""検証用の合成データ。実データが無くても基盤の正しさを確認できるようにする。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(closes, *, start="2024-01-01", spread=0.01, volume=1_000_000,
              gap=0.003) -> pd.DataFrame:
    """終値の列から、四本値と出来高を決定的に組み立てる。

    出来高は「上げた日は多く、下げた日は少ない」形にする。定数にすると
    `volume > SMA(volume, 20)` のような条件が永久に成立せず、テストにならない。

    始値には前日終値から小さなギャップを入れる。`open[t] == close[t-1]` にしてしまうと、
    「翌日始値で約定している」ことと「判定日の終値で約定している」ことを区別できず、
    ルックアヘッドの検査が意味を失う。
    """
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(closes))
    signs = np.array([1.0 if i % 2 else -1.0 for i in range(len(closes))])
    opens = np.concatenate([[closes[0]], closes[:-1] * (1 + gap * signs[1:])])
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    up = closes >= opens
    vol = np.where(up, volume * 1.4, volume * 0.7)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": vol.astype(float)},
        index=dates,
    )


def flat_then_breakout(n_flat=60, flat=100.0, n_up=40, step=2.0) -> pd.DataFrame:
    """横ばいのあと一本調子で上がる。ドンチャンブレイクが必ず1回起きる。"""
    wobble = [flat + (0.5 if i % 2 else -0.5) for i in range(n_flat)]
    up = [flat + step * (i + 1) for i in range(n_up)]
    return make_bars(wobble + up)


def flat_then_breakout_then_crash(n_flat=60, flat=100.0, n_up=15, step=2.0,
                                  n_down=30, drop=3.0) -> pd.DataFrame:
    wobble = [flat + (0.5 if i % 2 else -0.5) for i in range(n_flat)]
    up = [flat + step * (i + 1) for i in range(n_up)]
    peak = up[-1]
    # 一定額ずつ引くと価格が 0 以下になりうる（FR-108 の検証に引っかかる）。比率で下げる
    rate = drop / peak
    down = [peak * (1 - rate) ** (i + 1) for i in range(n_down)]
    return make_bars(wobble + up + down)


def rising_index(n: int, start="2024-01-01") -> pd.DataFrame:
    """常に200日線の上にある市場指数。レジームフィルタを通すために使う。"""
    lead = 330                                                   # SMA200 が埋まるだけの助走
    closes = np.linspace(1000.0, 1000.0 + n * 0.5, n + lead)
    end = pd.bdate_range(start, periods=n)[-1]
    dates = pd.bdate_range(end=end, periods=len(closes))          # 銘柄の最終日まで確実に覆う
    return pd.DataFrame({"close": closes}, index=dates)


def vcp_pattern(spread=0.005):
    """ST-04 が拾うべき形: 長い上昇 → 押し 12% → 7% → 3%（出来高も縮小）→ 出来高2倍で上放れ。

    戻り値: (bars, breakout_pos)。breakout_pos はピボットを終値で上抜けた日の位置。
    """
    closes, vols = [], []
    # 300本の上昇（SMA200 が上向き・52週安値の +30% 以上を満たす）
    for i in range(300):
        closes.append(100.0 * (1.0037 ** i)); vols.append(1.0)
    peak1 = closes[-1]

    def leg(start, end, n, vol):
        for k in range(1, n + 1):
            closes.append(start + (end - start) * k / n); vols.append(vol)

    leg(peak1, peak1 * 0.88, 12, 1.0)          # 押し1: -12%
    leg(closes[-1], peak1 * 1.02, 12, 0.9)     # 戻り → 新高値
    peak2 = closes[-1]
    leg(peak2, peak2 * 0.93, 10, 0.7)          # 押し2: -7%
    leg(closes[-1], peak2 * 1.01, 10, 0.8)     # 戻り
    peak3 = closes[-1]
    leg(peak3, peak3 * 0.97, 8, 0.45)          # 押し3: -3%
    leg(closes[-1], peak3 * 0.995, 6, 0.4)     # ピボット直下で横ばい
    breakout_pos = len(closes)
    closes.append(peak3 * 1.03); vols.append(2.2)   # 上放れ、出来高2.2倍
    for k in range(1, 15):                            # その後も上昇
        closes.append(closes[-1] * 1.004); vols.append(1.2)

    bars = make_bars(closes, spread=spread, gap=0.001)
    bars["volume"] = np.array(vols) * 1_000_000.0
    return bars, breakout_pos
