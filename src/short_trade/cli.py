"""コマンドライン。

  python -m short_trade setup                     ブラウザで認証情報を入力 → 接続確認 → 取得 → 基準線まで自動
  python -m short_trade selfcheck                 仕様ファイルの検査（ネットワーク不要）
  python -m short_trade smoke --strategy ST-06    合成データで基盤の健全性を確認（ネットワーク不要）
  python -m short_trade fetch  --start 2015-01-01 --end 2025-12-31 --codes 7203,6758
  python -m short_trade backtest --strategy ST-06 --start 2015-01-01 --end 2025-12-31
  python -m short_trade fetch --earnings           キャッシュ済み銘柄の決算発表予定日を取得（決算跨ぎ禁止に使う）
  python -m short_trade compare                    キャッシュからリスク率×スリッページの比較表を出す
  python -m short_trade correlate                  Phase 2 の戦略間の相関を測る（VR-045）
  python -m short_trade funnel --strategy ST-04    条件ファネル（取引が少ない戦略の診断）
  python -m short_trade fetch --universe --top 300 上場銘柄一覧＋時価総額で上位 N 銘柄を取得
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, UnsupportedSpec, run

DATA = None  # ROOT 定義後に設定
from .spec import assert_selected, load_all, load_common

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "catalog"
DATA = ROOT / "data" / "jquants"
EARNINGS_PATH = DATA / "earnings.parquet"
EARNINGS_PROGRESS = DATA / "earnings_progress.json"


def _earnings_coverage(data: dict, earnings) -> str:
    """決算日が何銘柄に紐づいたか。0 なら決算跨ぎ禁止は実質無効なので、必ず表示する。"""
    if earnings is None:
        return "無効（fetch --earnings で有効化）"
    n = sum(1 for sym in data if earnings.get(sym))
    return f"有効（{n}/{len(data)} 銘柄に決算日あり）" if n else f"**無効同然**（決算日が1銘柄にも一致しない。fetch --earnings を実行）"


def _load_cache(start=None, end=None, min_bars: int = 250):
    """キャッシュ済みの日足・指数・決算日を読む。backtest / compare 共通。"""
    from .jquants import to_bars, to_earnings_map

    files = sorted((DATA / "daily").glob("*.parquet"))
    if not files:
        raise SystemExit("キャッシュがありません。先に setup（または fetch）を実行してください")
    data, skipped = {}, 0
    for f in files:
        bars = to_bars(pd.read_parquet(f))
        if start:
            bars = bars.loc[start:]
        if end:
            bars = bars.loc[:end]
        if len(bars) > min_bars:
            data[f.stem] = bars
        else:
            skipped += 1
    index_path = DATA / "index.parquet"
    index = pd.read_parquet(index_path) if index_path.exists() else None
    earnings = to_earnings_map(pd.read_parquet(EARNINGS_PATH)) if EARNINGS_PATH.exists() else None
    return data, index, earnings, skipped


def _specs(strategy_id: str | None = None):
    specs = load_all(CATALOG)
    if strategy_id:
        specs = [s for s in specs if s.id == strategy_id]
        if not specs:
            raise SystemExit(f"戦略 {strategy_id} が見つかりません")
    return specs


def cmd_setup(args) -> int:
    from .setup_ui import serve
    serve(port=args.port, open_browser=not args.no_browser)
    return 0


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
        for w in res.warnings[:3]:
            print("  警告:", w)
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
    from .jquants import JQuantsClient, to_bars, to_index

    client = JQuantsClient()
    start = args.start or client.coverage()[0]      # 省略時は契約が覆う最古日から
    if args.universe:
        import json as _json

        info = client.listed_info()
        latest = client.daily_quotes(code="7203", start=None, end=None).tail(1)
        if latest.empty:
            raise SystemExit("最新日が特定できませんでした")
        latest_date = str(pd.to_datetime(latest["Date"].iloc[-1]).date())
        print(f"全銘柄の {latest_date} 時点の時価総額を取得しています…")
        snap = client.daily_quotes(on=latest_date)
        if "MktCap" not in snap.columns:
            raise SystemExit(f"MktCap 列がありません。実際の列: {list(snap.columns)}")
        merged = snap.merge(info[["Code", "CoName", "MktNm"]], on="Code", how="left")
        if args.market:
            hit = merged["MktNm"].astype(str).str.contains(args.market, na=False)
            if hit.any():
                merged = merged[hit]
            else:
                print(f"警告: 市場区分 '{args.market}' に一致する銘柄がありません。区分の例: {merged['MktNm'].dropna().unique()[:5]}")
        merged = merged.dropna(subset=["MktCap"]).sort_values("MktCap", ascending=False).head(args.top)
        codes = [str(c)[:4] if len(str(c)) == 5 and str(c).endswith("0") else str(c) for c in merged["Code"]]
        (DATA).mkdir(parents=True, exist_ok=True)
        (DATA / "universe.json").write_text(_json.dumps(
            {"as_of": latest_date, "market": args.market, "top": args.top,
             "codes": codes, "names": merged["CoName"].astype(str).tolist()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"時価総額上位 {len(codes)} 銘柄を選びました。日足を取得します（{start} から）…")
        ok = 0
        for i, code in enumerate(codes, 1):
            try:
                bars = to_bars(client.cached_daily_quotes(code, start=start, end=None))
                ok += 1 if len(bars) else 0
                if i % 25 == 0 or i == len(codes):
                    print(f"  [{i}/{len(codes)}] 取得済み {ok}")
            except Exception as e:
                print(f"  {code}: 失敗 {str(e)[:80]}")
        print(f"完了: {ok} 銘柄。universe.json に選定結果を保存しました")
        return 0
    if args.earnings:
        return _fetch_earnings(client, start, args.end)
    if args.index:
        out = ROOT / "data" / "jquants" / "index.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.index.lower() == "topix":
            raw = client.topix(start=start, end=args.end)
            if raw is None or raw.empty:
                raise SystemExit("TOPIX が取得できませんでした（プランにより不可。--index 1306 を試してください）")
            df = to_index(raw)
        else:
            df = to_bars(client.daily_quotes(code=args.index, start=start, end=args.end))[["close"]]
        df.sort_index().to_parquet(out)
        print(f"指数（{args.index}）{len(df)} 本を {out} に保存しました")
        return 0
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else []
    if not codes:
        info = client.listed_info()
        out = ROOT / "data" / "jquants" / "listed_info.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        info.to_parquet(out)
        print(f"上場銘柄一覧 {len(info)} 件を {out} に保存しました")
        return 0

    for code in codes:
        raw = client.cached_daily_quotes(code, start=start, end=args.end, refresh=args.refresh)
        bars = to_bars(raw)
        print(f"{code}: {len(bars)} 本  {bars.index.min()} 〜 {bars.index.max()}"
              if len(bars) else f"{code}: データなし")
    return 0


def _fetch_earnings(client, start: str, end: str | None, *, max_age_days: int = 1) -> int:
    """決算発表予定日を **キャッシュ済み銘柄ごとに** 取る（差分更新・再開可能）。

    経緯（docs/27 §27.1）: 最初は日ごとに10年ぶん問い合わせていたが、途中の1日の失敗で
    全体が落ちた。さらに取れたあとも、比較の結果が1件も変わらなかった（決算日が一致していない）。
    v2 の `/fins/earnings-date?code=` は1銘柄の全履歴を1回で返すので、こちらに切り替えた。
      - 対象は data/jquants/daily にある銘柄（＝バックテストの対象）
      - 銘柄ごとに取得日時を進捗ファイルに記録し、max_age_days より新しいものは取り直さない
      - 失敗した銘柄は次回に再試行する
    """
    import json
    from datetime import datetime, timedelta

    from .jquants import normalize_code

    codes = sorted(f.stem for f in (DATA / "daily").glob("*.parquet"))
    if not codes:
        raise SystemExit("キャッシュ済みの銘柄がありません。先に setup（または fetch）を実行してください")
    prog = json.loads(EARNINGS_PROGRESS.read_text()) if EARNINGS_PROGRESS.exists() else {}
    fetched: dict[str, str] = dict(prog.get("fetched", {}))
    cutoff = datetime.now() - timedelta(days=max_age_days)
    todo = [c for c in codes if c not in fetched or datetime.fromisoformat(fetched[c]) < cutoff]
    existing = pd.read_parquet(EARNINGS_PATH) if EARNINGS_PATH.exists() else pd.DataFrame()
    if not todo:
        print(f"決算発表予定日は {len(codes)} 銘柄ぶん取得済みです（{len(existing):,} 件）")
        return 0
    print(f"決算発表予定日を {len(todo)} 銘柄ぶん取得します（銘柄ごとに1回。数十秒〜数分）…")
    df, failed = client.earnings_dates_for_codes(todo, progress=print)
    now = datetime.now().isoformat(timespec="seconds")
    if not df.empty:
        df = df.copy()
        df["_code4"] = df["Code"].map(normalize_code)
        keep = existing[~existing["_code4"].isin(set(df["_code4"]))] if "_code4" in existing.columns else existing
        existing = pd.concat([keep, df]).reset_index(drop=True) if not keep.empty else df
        keys = [c for c in ("Code", "PubDate", "SchDate", "FYE", "FQName") if c in existing.columns]
        existing = existing.drop_duplicates(subset=keys or None).reset_index(drop=True)
        EARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(EARNINGS_PATH)
    for c in todo:
        if c not in failed:
            fetched[c] = now
    EARNINGS_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    EARNINGS_PROGRESS.write_text(json.dumps({"fetched": fetched, "failed": failed, "rows": int(len(existing))},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
    got = len(set(existing["_code4"]) & set(codes)) if "_code4" in existing.columns else 0
    if failed:
        print(f"取得できなかった銘柄が {len(failed)} 件あります（{failed[:5]}）。次回の実行で再試行します")
    print(f"決算発表予定日 {len(existing):,} 件を保存しました（{got}/{len(codes)} 銘柄に日付あり）")
    return 0

def cmd_backtest(args) -> int:
    data, index, earnings, skipped = _load_cache(args.start, args.end)
    if index is None:
        print("警告: 指数データがありません。レジームフィルタが評価できません（docs/19 §19.4 の代用案を参照）")
    if earnings is None:
        print("注意: 決算日データがありません。`fetch --earnings` を実行すると決算跨ぎ禁止（RM-001d）が有効になります")
    if skipped:
        print(f"注意: 履歴が短い {skipped} 銘柄を除外しました")

    for spec in _specs(args.strategy):
        overrides = {"initial_equity": args.equity, "earnings_dates": earnings}
        if args.slippage is not None:
            overrides["slippage_pct"] = args.slippage
        if args.risk_pct is not None:
            overrides["risk_pct_override"] = args.risk_pct
        print(f"\n=== {spec.id} {spec.name} ===")
        try:
            res = run(spec, data, index=index,
                      config=BacktestConfig.from_common(spec.common, **overrides))
        except UnsupportedSpec as e:
            print(f"  未実装: {e}")
            continue
        for k, v in res.metrics().items():
            print(f"  {k:20} {v:>12.2f}" if isinstance(v, float) else f"  {k:20} {v:>12}")
        if res.rejections:
            print("  却下:", res.rejections)
        for w in res.warnings[:5]:
            print("  警告:", w)
    return 0


def cmd_compare(args) -> int:
    """キャッシュ済みデータで、リスク率 × スリッページの組合せを一覧にする。

    5万円ではリスク率1%だとシグナルの大半が「1株未満」で捨てられる（docs/24）。
    1% と 2% を並べて、取引数と最大DDのトレードオフを見て決めるための表。
    """
    import json

    data, index, earnings, skipped = _load_cache(args.start, args.end)
    if index is None:
        raise SystemExit("指数データがありません。setup を先に実行してください")
    risk_levels = [float(x) for x in args.risk.split(",")]
    slips = [float(x) for x in args.slippage_levels.split(",")]
    policies = [x.strip() for x in (getattr(args, "earnings_policies", None) or "").split(",") if x.strip()]
    matched = sum(1 for sym in data if (earnings or {}).get(sym))
    report: dict = {"equity": args.equity, "symbols": len(data), "skipped": skipped,
                    "earnings_applied": earnings is not None, "earnings_symbols_matched": matched,
                    "results": {}}
    print(f"銘柄 {len(data)} / 資金 {args.equity:,.0f}円 / 決算跨ぎ禁止: {_earnings_coverage(data, earnings)}")
    specs = _specs(args.strategy)
    if not args.strategy and getattr(args, "phase", None):
        specs = [s for s in specs if s.phase == args.phase]
    for spec in specs:
        rows = []
        print(f"\n=== {spec.id} {spec.name} ===")
        print(f"  {'リスク%':>6} {'滑り%':>5} {'取引':>5} {'勝率':>6} {'PF':>6} {'平均R':>6} {'最大DD':>7} {'連敗':>4} {'総損益':>9}  1株未満で却下")
        # 格子: リスク率 × スリッページ（決算方針は仕様の既定）。
        # 加えて、決算方針の比較を「最も現実に近い滑り（最大値）」でだけ行う（docs/28 §28.1）
        grid = [(r, sl, None) for r in risk_levels for sl in slips]
        if earnings is not None and policies:
            grid += [(r, max(slips), pol) for r in risk_levels for pol in policies]
        for r, sl, pol in grid:
            over = {"earnings_policy": pol} if pol else {}
            cfg = BacktestConfig.from_common(spec.common, initial_equity=args.equity,
                                             slippage_pct=sl, risk_pct_override=r,
                                             earnings_dates=earnings, **over)
            try:
                res = run(spec, data, index=index, config=cfg)
            except UnsupportedSpec as e:
                print(f"  未実装: {e}")
                rows = None
                break
            m = res.metrics()
            too_small = sum(v for k, v in res.rejections.items() if "1株未満" in k)
            row = {"risk_pct": r, "slippage_pct": sl, "earnings_policy": pol or cfg.earnings_policy,
                   **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                   "rejected_too_small": too_small, "rejections": res.rejections}
            rows.append(row)
            tag = f"  決算方針={pol}" if pol else ""
            print(f"  {r:>6.1f} {sl:>5.2f} {m.get('取引数', 0):>5} {m.get('勝率', 0):>5.1f}% {m.get('プロフィットファクタ', 0):>6.2f} "
                  f"{m.get('平均R', 0):>6.2f} {m.get('最大DD', 0):>6.1f}% {m.get('最大連敗', 0):>4} {m.get('総損益', 0):>9,.0f}  {too_small}{tag}")
        if rows:
            report["results"][spec.id] = rows
    out = ROOT / "data" / "compare_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n比較表を {out.relative_to(ROOT)} に書き出しました。この内容を報告してください（認証情報は含まれません）")
    return 0


def cmd_funnel(args) -> int:
    """条件ファネル: 各条件で候補がどれだけ残るかを数える（取引が少ない戦略の診断）。"""
    import json

    from .diagnose import format_funnel, funnel

    data, index, earnings, skipped = _load_cache(args.start, args.end)
    specs = _specs(args.strategy)
    if not args.strategy and getattr(args, "phase", None):
        specs = [s for s in specs if s.phase == args.phase]
    report = {"symbols": len(data), "skipped": skipped, "funnels": {}}
    print(f"銘柄 {len(data)}（250本未満で除外 {skipped}）")
    for spec in specs:
        try:
            f = funnel(spec, data, index=index, config=BacktestConfig.from_common(spec.common, earnings_dates=earnings))
        except UnsupportedSpec as e:
            print(f"  {spec.id}: 未実装 {e}")
            continue
        print()
        print(format_funnel(f))
        report["funnels"][spec.id] = f.to_dict()
    out = ROOT / "data" / "funnel_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out.relative_to(ROOT)} に書き出しました")
    return 0


def cmd_correlate(args) -> int:
    from .correlate import format_table, measure, save

    data, index, earnings, skipped = _load_cache(args.start, args.end)
    if index is None:
        raise SystemExit("指数データがありません。setup を先に実行してください")
    specs = _specs()
    if args.phase:
        specs = [s for s in specs if s.phase == args.phase]
    if args.strategies:
        want = {x.strip() for x in args.strategies.split(",")}
        specs = [s for s in specs if s.id in want]
    print(f"銘柄 {len(data)} / 相関測定用の資金 {args.equity:,.0f}円（サイズ制約をほぼ外して構造を見る）"
          f" / 決算跨ぎ禁止: {_earnings_coverage(data, earnings)}")
    rep = measure(specs, data, index=index, equity=args.equity, earnings=earnings)
    rep.earnings_symbols_matched = sum(1 for sym in data if (earnings or {}).get(sym))
    print("\n=== 戦略間の相関（docs/18 §18.4 の事前予想との照合） ===")
    print(format_table(rep))
    out = ROOT / "data" / "correlation_report.json"
    save(rep, out)
    flags = rep.flags()
    if flags:
        print("\n0.7 を超えたペアは同一因子として扱い、代表を1本に絞る（VR-045 / DEC-062）:")
        for a, b, v in flags:
            print(f"  {a} と {b}: {v:.2f}")
    print(f"\n{out.relative_to(ROOT)} に書き出しました。この内容を報告してください")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="short_trade", description="短期売買ツール")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("setup", help="ブラウザで認証情報を入力し、接続確認〜基準線まで自動実行")
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--no-browser", action="store_true")
    st.set_defaults(func=cmd_setup)

    sub.add_parser("selfcheck", help="仕様ファイルの検査").set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("smoke", help="合成データで基盤の健全性を確認")
    s.add_argument("--strategy")
    s.set_defaults(func=cmd_smoke)

    f = sub.add_parser("fetch", help="J-Quants からデータを取得")
    f.add_argument("--codes", help="カンマ区切りの銘柄コード。省略すると上場銘柄一覧を取得")
    f.add_argument("--index", help="指数を取得して index.parquet に保存。'topix' または代用ETFのコード（例: 1306）")
    f.add_argument("--earnings", action="store_true", help="キャッシュ済み銘柄の決算発表予定日を取得して earnings.parquet に保存（差分更新）")
    f.add_argument("--universe", action="store_true", help="上場銘柄一覧＋時価総額で上位 N 銘柄の日足を取得")
    f.add_argument("--market", default="プライム", help="--universe の市場区分（部分一致）。空文字で全市場")
    f.add_argument("--top", type=int, default=300, help="--universe で選ぶ銘柄数（時価総額上位）")
    f.add_argument("--start", default=None, help="省略すると契約が覆う最古日から取得します")
    f.add_argument("--end", default=None)
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="キャッシュ済みデータでバックテスト")
    b.add_argument("--strategy")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--equity", type=float, default=50_000)
    b.add_argument("--slippage", type=float, default=None,
                   help="片道スリッページ%%。VR-016: 0（楽観）と 0.1（保守）の両方で実行して比較する")
    b.add_argument("--risk-pct", type=float, default=None, help="1トレードのリスク率%%（仕様の値を上書き）")

    c = sub.add_parser("compare", help="キャッシュからリスク率×スリッページの比較表を出す")
    c.add_argument("--strategy", default=None, help="1本だけ比較する。省略すると --phase の全戦略")
    c.add_argument("--phase", type=int, default=2, help="対象フェーズ（既定: Phase 2）")
    c.add_argument("--risk", default="1.0,2.0", help="カンマ区切りのリスク率%%")
    c.add_argument("--slippage-levels", default="0.0,0.1", help="カンマ区切りの片道スリッページ%%")
    c.add_argument("--earnings-policies", default="entry_only,cushion",
                   help="決算跨ぎ方針の比較（最大スリッページでのみ実行）。空文字で省略")
    c.add_argument("--equity", type=float, default=50_000)
    c.add_argument("--start", default=None)
    c.add_argument("--end", default=None)
    c.set_defaults(func=cmd_compare)

    fu = sub.add_parser("funnel", help="条件ファネル: 各条件で候補がどれだけ残るかを数える")
    fu.add_argument("--strategy", default=None)
    fu.add_argument("--phase", type=int, default=2)
    fu.add_argument("--start", default=None)
    fu.add_argument("--end", default=None)
    fu.set_defaults(func=cmd_funnel)

    r = sub.add_parser("correlate", help="戦略間の相関を測る（VR-045）")
    r.add_argument("--phase", type=int, default=2, help="対象フェーズ（既定: Phase 2）")
    r.add_argument("--strategies", default=None, help="カンマ区切りで明示（--phase より優先）")
    r.add_argument("--equity", type=float, default=10_000_000)
    r.add_argument("--start", default=None)
    r.add_argument("--end", default=None)
    r.set_defaults(func=cmd_correlate)
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    return args.func(args)
