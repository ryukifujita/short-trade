"""セットアップ画面のエンドツーエンド検査。J-Quants は偽クライアントに差し替える。"""
import json
import os
import threading
import time
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_trade import jquants, setup_ui
from synthetic import make_bars


def _jq_frame(n=1500, start="2016-01-04"):
    """J-Quants の daily_quotes と同じ列名で返す（to_bars の想定スキーマの検査を兼ねる）。"""
    bars = make_bars([100 * (1.0004 ** i) * (1 + 0.01 * np.sin(i / 7)) for i in range(n)], start=start)
    return pd.DataFrame({
        "Date": bars.index.strftime("%Y-%m-%d"),
        "Code": "72030",
        "Open": bars["open"], "High": bars["high"], "Low": bars["low"], "Close": bars["close"],
        "Volume": bars["volume"],
        "AdjustmentOpen": bars["open"], "AdjustmentHigh": bars["high"], "AdjustmentLow": bars["low"],
        "AdjustmentClose": bars["close"], "AdjustmentVolume": bars["volume"],
    }).reset_index(drop=True)


class FakeClient:
    def __init__(self, credentials=None, **kw):
        self.credentials = credentials
        if not (credentials and (credentials.refresh_token or credentials.mail_address)):
            raise jquants.JQuantsError("認証情報が見つかりません")
        self.cache_dir = Path(kw.get("cache_dir", "data/jquants"))

    @property
    def id_token(self):
        return "fake"

    def listed_info(self, on=None):
        return pd.DataFrame({"Code": ["72030", "67580"], "CompanyName": ["A", "B"], "MarketCode": ["0111", "0111"]})

    def daily_quotes(self, *, code=None, on=None, start=None, end=None):
        return _jq_frame()

    def cached_daily_quotes(self, code, *, start, end, refresh=False):
        df = _jq_frame()
        path = self.cache_dir / "daily" / f"{code}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    def topix(self, *, start=None, end=None):
        raise jquants.JQuantsError("HTTP 403 (無料プランでは不可)")

    def announcement(self):
        return pd.DataFrame({"Date": ["2026-09-11"], "Code": ["72030"]})


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(jquants, "JQuantsClient", FakeClient)
    monkeypatch.setattr(setup_ui, "ROOT", tmp_path)
    monkeypatch.setattr(setup_ui, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(setup_ui, "REPORT_PATH", tmp_path / "data" / "setup_report.json")
    monkeypatch.setattr(setup_ui, "SEED_CODES", ["7203", "6758"])
    # catalog は本物を使う
    real_root = Path(__file__).resolve().parents[1]
    (tmp_path / "catalog").symlink_to(real_root / "catalog")
    setup_ui._state.update({"phase": "idle", "log": [], "report": None, "error": None})
    server = HTTPServer(("127.0.0.1", 0), setup_ui._Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()


def _post(url, data: dict):
    body = "&".join(f"{k}={v}" for k, v in data.items()).encode()
    req = urllib.request.Request(url, data=body, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    try:
        return opener.open(req)
    except urllib.error.HTTPError as e:
        return e


def test_form_is_served(sandbox):
    url, _ = sandbox
    html = urllib.request.urlopen(url).read().decode()
    assert "JQUANTS_REFRESH_TOKEN" in html and "接続を確認" in html


def test_empty_submission_is_rejected(sandbox):
    url, root = sandbox
    r = _post(url + "/start", {"refresh_token": "", "mail": "", "password": ""})
    assert r.status == 200 and "どちらかを入力" in r.read().decode()
    assert not (root / ".env").exists()


def test_full_pipeline_with_fake_client(sandbox):
    url, root = sandbox
    r = _post(url + "/start", {"refresh_token": "TESTTOKEN123", "mail": "", "password": ""})
    assert r.status == 303
    deadline = time.time() + 60
    status = {}
    while time.time() < deadline:
        status = json.loads(urllib.request.urlopen(url + "/status").read())
        if status["phase"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert status["phase"] == "done", "\n".join(status.get("log", []))

    # .env に保存され、権限が所有者のみ
    env = (root / ".env").read_text()
    assert "JQUANTS_REFRESH_TOKEN=TESTTOKEN123" in env
    if os.name != "nt":
        assert oct(os.stat(root / ".env").st_mode & 0o777) == "0o600"

    # レポートに U-1 / U-3 / 基準線が入り、認証情報は入っていない
    report = json.loads((root / "data" / "setup_report.json").read_text())
    assert report["U-1_earliest_date"] == "2016-01-04"
    assert report["U-3_topix_available"] is False and "1306" in report["index_used"]
    assert "slippage_0.0" in report["baseline"] and "slippage_0.1" in report["baseline"]
    assert "TESTTOKEN123" not in json.dumps(report)
    assert "TESTTOKEN123" not in "\n".join(status["log"])
    assert "完了" in status["result_html"]
