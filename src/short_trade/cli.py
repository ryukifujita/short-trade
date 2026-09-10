"""コマンドライン。

  python -m short_trade selfcheck                 仕様ファイルの検査（ネットワーク不要）
  python -m short_trade smoke --strategy ST-06    合成データで基盤の健全性を確認（ネットワーク不要）
  python -m short_trade fetch  --start 2015-01-01 --end 2025-12-31 --codes 7203,6758
  python -m short_trade backtest --strategy ST-06 --start 2015-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, UnsupportedSpec, run
from .spec import assert_selected, load_all, load_common

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "catalog"


def _specs(strategy_id: str | None = None):
    specs = load_all(CATALOG)
    if strategy_id:
        specs = [s for s in specs if s.id == strategy_id]
        if not specs:
            raise SystemExit(f"戦略 {strategy_id} が見つかりません")
    return specs


def cmd_selfcheck(args) -> int:
    specs = _specs()
    assert_selected(CATALOG, [s.id for s in specs])
    total_dof = sum(s.degrees_of_freedom for s in specs)
    print(f"仕様ファイル {len(specs)} 本を読み込みました")
    for s in specs:
        print(f"  {s.id}  {s.name}")
        print(f"        因子={s.factor} Phase={s.phase} 自由度={s.degrees_of_freedom} "
              f"リスク={s.risk_pct}% 条件数={len(s.entry_conditions)}")
    print(f"\n自由度の合計: {total_dof}（docs/18 §18.5 の記録と一致すること）")
    print("FR-C07: 全戦略が代表手法として選抜済みであることを確認しました")
    return 0


def cmd_smoke(args) -> int:
    """合成データで動かし、基盤が壊れていないことを確認する。実データの成績とは無関係。"""
    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic import flat_then_breakout_then_crash, rising_index  # noqa: E402

    df = flat_then_breakout_then_crash(n_flat=250, n_up=40, n_down=60)
    idx = rising_index(len(df), start=str(df.index[0].date()))
    for spec in _specs(args.strategy):
        print(f"\n=== {spec.id} {spec.name} （合成データ） ===")
        try:
            res = run(spec, {"TEST": df}, index=idx,
                      config=BacktestConfig.from_common(spec.common, initial_equity=50_000))
        except UnsupportedSpec as e:
            print(f"  未実装: {e}")
            continue
        m = res.metrics()
        if m["取引数"] == 0:
            print("  シグナルなし。却下理由:", res.rejections or "（条件不成立）")
            continue
        for k, v in m.items():
            print(f"  {k:20} {v:>12.2f}" if isinstance(v, float) else f"  {k:20} {v:>12}")
        if res.rejections:
            print("  却下:", res.rejections)
    return 0


def cmd_fetch(args) -> int:
    from .jquants import JQuantsClient, to_bars

    client = JQuantsClient()
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else []
    if not codes:
        info = client.listed_info()
        out = ROOT / "data" / "jquants" / "listed_info.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        info.to_parquet(out)
        print(f"上場銘柄一覧 {len(info)} 件を {out} に保存しました")
        return 0

    for code in codes:
        raw = client.cached_daily_quotes(code, start=args.start, end=args.end,
                                         refresh=args.refresh)
        bars = to_bars(raw)
        print(f"{code}: {len(bars)} 本  {bars.index.min()} 〜 {bars.index.max()}"
              if len(bars) else f"{code}: データなし")
    return 0


def cmd_backtest(args) -> int:
    from .jquants import to_bars

    cache = ROOT / "data" / "jquants" / "daily"
    files = sorted(cache.glob("*.parquet"))
    if not files:
        raise SystemExit(
            "キャッシュがありません。先に `python -m short_trade fetch` を実行してください"
        )
    data = {}
    for f in files:
        bars = to_bars(pd.read_parquet(f))
        if args.start:
            bars = bars.loc[args.start:]
        if args.end:
            bars = bars.loc[:args.end]
        if len(bars) > 250:
            data[f.stem] = bars
    index_path = ROOT / "data" / "jquants" / "index.parquet"
    index = pd.read_parquet(index_path) if index_path.exists() else None
    if index is None:
        print("警告: 指数データがありません。レジームフィルタが評価できません（docs/19 §19.4 の代用案を参照）")

    for spec in _specs(args.strategy):
        res = run(spec, data, index=index,
                  config=BacktestConfig.from_common(spec.common,
                                                    initial_equity=args.equity))
        print(f"\n=== {spec.id} {spec.name} ===")
        for k, v in res.metrics().items():
            print(f"  {k:20} {v:>12.2f}" if isinstance(v, float) else f"  {k:20} {v:>12}")
        if res.rejections:
            print("  却下:", res.rejections)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="short_trade", description="短期売買ツール")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="仕様ファイルの検査").set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("smoke", help="合成データで基盤の健全性を確認")
    s.add_argument("--strategy")
    s.set_defaults(func=cmd_smoke)

    f = sub.add_parser("fetch", help="J-Quants からデータを取得")
    f.add_argument("--codes", help="カンマ区切りの銘柄コード。省略すると上場銘柄一覧を取得")
    f.add_argument("--start", default="2015-01-01")
    f.add_argument("--end", default=None)
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="キャッシュ済みデータでバックテスト")
    b.add_argument("--strategy")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--equity", type=float, default=50_000)
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    return args.func(args)
