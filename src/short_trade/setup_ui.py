"""セットアップ画面（ローカル専用）。

    python -m short_trade setup

ブラウザに入力画面を開き、J-Quants の認証情報を受け取って次を全自動で行う:
  1. 接続確認（APIキー → 上場銘柄一覧 → 日足の最古日 → TOPIX の可否 → 決算発表日の可否）
  2. 認証情報を .env に保存（リポジトリにはコミットされない。.gitignore 済み）
  3. 指数と代表銘柄の日足を取得してローカルに保存
  4. ST-06（ドンチャン）の基準線バックテストをスリッページ 0 / 0.1% で実行
  5. 結果を data/setup_report.json に書き出し、画面に表示

このサーバは 127.0.0.1 にしか bind しない。外部からは届かない。
認証情報はログにも画面にも出さない（NFR-021）。
"""
from __future__ import annotations

import html
import json
import os
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
REPORT_PATH = ROOT / "data" / "setup_report.json"

# 基準線（ST-06）を回すための代表銘柄。成績を見るためではなく、基盤の動作確認用。
SEED_CODES = [
    "7203", "6758", "9432", "8058", "4063", "6098", "6501", "8035", "9984", "7974",
    "6861", "8306", "4502", "6902", "7741", "4568", "6367", "9433", "8031", "6594",
    "4519", "6954", "7267", "8766", "6857", "4661", "9983", "2914", "8001", "3382",
]

_state: dict = {"phase": "idle", "log": [], "report": None, "error": None}
_lock = threading.Lock()


def _log(msg: str) -> None:
    with _lock:
        _state["log"].append(f"{datetime.now():%H:%M:%S} {msg}")


def _save_env(values: dict[str, str]) -> None:
    """認証情報を .env に保存する（既存の他キーは保持）。パーミッションは所有者のみ。"""
    existing: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    for k in ("JQUANTS_API_KEY", "JQUANTS_REFRESH_TOKEN", "JQUANTS_MAIL_ADDRESS", "JQUANTS_PASSWORD"):
        existing.pop(k, None)   # 旧v1の残骸も消す
    existing.update({k: v for k, v in values.items() if v})
    ENV_PATH.write_text("".join(f"{k}={v}\n" for k, v in existing.items()), encoding="utf-8")
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------- パイプライン

def _pipeline(creds: dict[str, str]) -> None:
    """接続確認 → 保存 → 取得 → 基準線バックテスト。別スレッドで走る。"""
    import pandas as pd

    from .backtest import BacktestConfig, run
    from .jquants import Credentials, JQuantsClient, to_bars, to_index
    from .spec import load_all

    report: dict = {"checked_at": datetime.now().isoformat(timespec="seconds"), "unverified": {}}
    try:
        # ---- 1) 接続確認 ----
        with _lock:
            _state["phase"] = "verify"
        _log("APIキーで接続しています…")
        client = JQuantsClient(Credentials(api_key=creds["api_key"]), cache_dir=ROOT / "data" / "jquants")

        _log("上場銘柄一覧を取得しています…")
        info = client.listed_info()
        _log("接続OK")
        report["listed_count"] = int(len(info))
        report["listed_columns"] = list(map(str, info.columns))
        _log(f"上場銘柄一覧: {len(info)} 件")

        _log("契約が覆う期間を調べています…")
        cov_start, cov_end = client.coverage()
        report["U-1_coverage_start"] = cov_start
        report["U-1_coverage_end"] = cov_end
        _log(f"契約範囲: {cov_start} 〜 {cov_end or '最新'}")

        _log("日足を取得できるか試しています（7203）…")
        raw = client.daily_quotes(code="7203", start=cov_start)
        bars = to_bars(raw)
        report["daily_columns"] = list(map(str, raw.columns))
        report["U-1_earliest_date"] = str(bars.index.min().date()) if len(bars) else None
        report["U-1_latest_date"] = str(bars.index.max().date()) if len(bars) else None
        report["U-1_years"] = round(len(bars) / 245, 1) if len(bars) else 0
        _log(f"日足: {report['U-1_earliest_date']} 〜 {report['U-1_latest_date']}（約 {report['U-1_years']} 年）")

        _log("TOPIX 指数が取れるか試しています…")
        topix = None
        try:
            topix = client.topix(start=cov_start)
            report["U-3_topix_available"] = bool(len(topix))
        except Exception as e:  # 無料プランでは不可の可能性（docs/19 U-3）
            report["U-3_topix_available"] = False
            report["U-3_error"] = str(e)[:200]
        _log("TOPIX: " + ("取得できました" if report["U-3_topix_available"] else "取得できません → 1306 で代用します"))

        _log("決算発表予定が取れるか試しています…")
        try:
            today = datetime.now().date()
            ann = client.earnings_dates(start=str(today - pd.Timedelta(days=7)), end=str(today))
            report["U-5_earnings_date_available"] = bool(len(ann))
            report["U-5_rows"] = int(len(ann))
            report["U-5_columns"] = list(map(str, ann.columns))[:12]
        except Exception as e:
            report["U-5_earnings_date_available"] = False
            report["U-5_error"] = str(e)[:200]

        # ---- 2) 保存 ----
        _save_env({"JQUANTS_API_KEY": creds.get("api_key") or ""})
        _log(f"認証情報を {ENV_PATH.name} に保存しました（gitignore 済み）")

        # ---- 3) 取得 ----
        with _lock:
            _state["phase"] = "fetch"
        index_out = ROOT / "data" / "jquants" / "index.parquet"
        index_out.parent.mkdir(parents=True, exist_ok=True)
        if report["U-3_topix_available"]:
            df = to_index(topix)
            report["index_used"] = "TOPIX"
        else:
            df = to_bars(client.daily_quotes(code="1306", start=cov_start))[["close"]]
            report["index_used"] = "1306（TOPIX連動ETFで代用）"
        df.sort_index().to_parquet(index_out)
        _log(f"指数（{report['index_used']}）を保存しました: {len(df)} 本")

        start = report["U-1_coverage_start"] or report["U-1_earliest_date"] or "2016-01-01"
        got = 0
        for i, code in enumerate(SEED_CODES, 1):
            try:
                b = to_bars(client.cached_daily_quotes(code, start=start, end=None))
                got += 1 if len(b) else 0
                _log(f"[{i}/{len(SEED_CODES)}] {code}: {len(b)} 本")
            except Exception as e:
                _log(f"[{i}/{len(SEED_CODES)}] {code}: 失敗 {str(e)[:80]}")
        report["fetched_symbols"] = got

        # ---- 4) 基準線バックテスト ----
        with _lock:
            _state["phase"] = "backtest"
        data = {}
        for f in sorted((ROOT / "data" / "jquants" / "daily").glob("*.parquet")):
            b = to_bars(pd.read_parquet(f))
            if len(b) > 250:
                data[f.stem] = b
        index = pd.read_parquet(index_out)
        spec = next(s for s in load_all(ROOT / "catalog") if s.id == "ST-06")
        report["baseline"] = {}
        for slip in (0.0, 0.1):
            _log(f"ST-06 バックテスト（スリッページ {slip}%）…")
            res = run(spec, data, index=index,
                      config=BacktestConfig.from_common(spec.common, initial_equity=50_000, slippage_pct=slip))
            m = res.metrics()
            report["baseline"][f"slippage_{slip}"] = {
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                "rejections": res.rejections,
                "warnings": res.warnings[:5],
            }
            _log(f"  取引数 {m.get('取引数')} / 平均R {m.get('平均R', 0):.2f} / 最大DD {m.get('最大DD', 0):.1f}%")

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"レポートを {REPORT_PATH.relative_to(ROOT)} に書き出しました")
        with _lock:
            _state["report"] = report
            _state["phase"] = "done"
    except Exception as e:
        with _lock:
            _state["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            _state["phase"] = "error"
        _log("エラー: " + _state["error"])
        _log(traceback.format_exc().splitlines()[-1])


# ---------------------------------------------------------------- 画面

_CSS = """
body{font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#101a1c;line-height:1.7}
h1{font-size:22px;margin:0 0 6px}p.lead{color:#5c6e72;margin:0 0 24px}
fieldset{border:1px solid #d9e2e3;border-radius:10px;padding:16px 18px;margin:0 0 16px}
legend{font-weight:700;padding:0 6px}label{display:block;font-size:13px;color:#33474b;margin:10px 0 4px}
input{width:100%;box-sizing:border-box;font:inherit;font-size:15px;padding:10px 12px;border:1.5px solid #bccbcd;border-radius:8px}
button{font:inherit;font-size:15px;font-weight:700;padding:12px 20px;border:none;border-radius:8px;background:#0e6e6e;color:#fff;cursor:pointer}
.note{font-size:13px;color:#5c6e72;background:#f4f7f7;border-radius:8px;padding:10px 12px}
pre{background:#101a1c;color:#e7eeef;padding:14px;border-radius:8px;font-size:12.5px;max-height:360px;overflow:auto;white-space:pre-wrap}
.ok{color:#0a5252;font-weight:700}.ng{color:#c43d2e;font-weight:700}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #d9e2e3;padding:7px 8px;text-align:left}
"""

_FORM = f"""<!doctype html><meta charset="utf-8"><title>short-trade セットアップ</title><style>{_CSS}</style>
<h1>J-Quants の接続設定</h1>
<p class="lead">APIキーを貼るだけです。接続確認 → 保存 → データ取得 → 基準線バックテストまで自動で進みます。</p>
<form method="post" action="/start">
<fieldset><legend>APIキー</legend>
<label>JQUANTS_API_KEY</label>
<input name="api_key" autocomplete="off" placeholder="ダッシュボードの「APIキー」に表示されている文字列">
<div class="note">
J-Quants にログイン → ダッシュボード → <b>APIキー</b> の欄に表示されているものです。<br>
他のツールで使っているものをそのまま貼れます。<b>有効期限はありません</b>（1週間で切れるのは旧方式のトークンで、そちらは廃止済みです）。
</div>
</fieldset>
<button type="submit">接続を確認して、続きを自動で進める</button>
<p class="note">保存先はこのフォルダの <code>.env</code>（Git には入りません）。入力内容はこのPCの外に送られません（J-Quants への接続を除く）。</p>
</form>"""


def _progress_page() -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>short-trade セットアップ</title><style>{_CSS}</style>
<h1>実行中…</h1><p class="lead" id="phase"></p><pre id="log"></pre><div id="result"></div>
<script>
async function tick(){{
  const r = await fetch('/status'); const s = await r.json();
  document.getElementById('log').textContent = s.log.join('\\n');
  const names = {{idle:'待機', verify:'1/3 接続確認', fetch:'2/3 データ取得', backtest:'3/3 基準線バックテスト', done:'完了', error:'エラー'}};
  document.getElementById('phase').textContent = names[s.phase] || s.phase;
  if (s.phase === 'done') {{ document.getElementById('result').innerHTML = s.result_html; return; }}
  if (s.phase === 'error') {{ document.getElementById('result').innerHTML =
     '<p class="ng">失敗しました。上のログの最後の行をそのまま貼って報告してください。</p><p><a href="/">入力画面へ戻る</a></p>'; return; }}
  setTimeout(tick, 1500);
}}
tick();
</script>"""


def _result_html(report: dict) -> str:
    def yn(v):
        return '<span class="ok">取得できる</span>' if v else '<span class="ng">取得できない</span>'
    rows = [
        ("上場銘柄一覧", f"{report.get('listed_count')} 件"),
        ("契約が覆う期間", f"{report.get('U-1_coverage_start')} 〜 {report.get('U-1_coverage_end') or '最新'}"),
        ("U-1 日足の最古日", f"{report.get('U-1_earliest_date')}（約 {report.get('U-1_years')} 年）"),
        ("U-1 日足の最新日", f"{report.get('U-1_latest_date')}（無料プランは12週遅延）"),
        ("U-3 TOPIX 指数", yn(report.get("U-3_topix_available")) + f" → 使用: {report.get('index_used')}"),
        ("U-5 決算発表予定日", yn(report.get("U-5_earnings_date_available"))),
        ("取得した銘柄", f"{report.get('fetched_symbols')} / {len(SEED_CODES)}"),
    ]
    tbl = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    bt = ""
    for key, m in (report.get("baseline") or {}).items():
        bt += f"<h3>ST-06 基準線（{html.escape(key)}）</h3><table>" + "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in m.items() if k not in ("rejections", "warnings")
        ) + "</table>"
        if m.get("rejections"):
            bt += f"<p class='note'>却下: {html.escape(json.dumps(m['rejections'], ensure_ascii=False))}</p>"
    years = report.get("U-1_years") or 0
    verdict = ('<p class="ok">遡れる期間が5年以上あります。一括取得（有料プラン1か月契約）は不要です。</p>'
               if years >= 5 else
               '<p class="ng">遡れる期間が5年未満です。docs/19 §19.4 の一括取得案（有料プランを1か月だけ契約）を検討してください。</p>')
    return (f"<h2>完了</h2><table>{tbl}</table>{verdict}{bt}"
            f"<p class='note'>この内容は <code>data/setup_report.json</code> に保存されました。"
            f"<b>そのファイルの中身を貼って報告してください</b>（認証情報は含まれていません）。</p>")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 標準のアクセスログを黙らせる（認証情報が混ざらないように）
        return

    def _send(self, body: str, status: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            with _lock:
                payload = {
                    "phase": _state["phase"], "log": _state["log"][-200:],
                    "result_html": _result_html(_state["report"]) if _state["report"] else "",
                }
            self._send(json.dumps(payload, ensure_ascii=False), ctype="application/json; charset=utf-8")
        elif self.path == "/progress":
            self._send(_progress_page())
        else:
            self._send(_FORM)

    def do_POST(self):
        if self.path != "/start":
            self._send("not found", 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        creds = {"api_key": form.get("api_key", [""])[0].strip() or None}
        if not creds["api_key"]:
            self._send(_FORM.replace("<form", '<p class="ng">APIキーを入力してください。</p><form'))
            return
        with _lock:
            if _state["phase"] in ("verify", "fetch", "backtest"):
                self._send("実行中です", 409)
                return
            _state.update({"phase": "verify", "log": [], "report": None, "error": None})
        # .env の保存はパイプライン側（認証成功後）で行う
        threading.Thread(target=_pipeline, args=(dict(creds),), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/progress")
        self.end_headers()


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"セットアップ画面を開きます: {url}")
    print("終了するには Ctrl+C")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
