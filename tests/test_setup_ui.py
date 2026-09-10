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
    """J-Quants v2 の /equities/bars/daily と同じ列名で返す。

    列名は jquantsapi.constants.EQ_BARS_DAILY_COLUMNS_V2 に合わせている。
    to_bars がこのスキーマを正しく読めるかの検査を兼ねる。
    """
    bars = make_bars([100 * (1.0004 ** i) * (1 + 0.01 * np.sin(i / 7)) for i in range(n)], start=start)
    return pd.DataFrame({
        "Date": bars.index.strftime("%Y-%m-%d"),
        "Code": "72030",
        "O": bars["open"], "H": bars["high"], "L": bars["low"], "C": bars["close"],
        "Vo": bars["volume"], "Va": bars["close"] * bars["volume"], "AdjFactor": 1.0,
        "AdjO": bars["open"], "AdjH": bars["high"], "AdjL": bars["low"],
        "AdjC": bars["close"], "AdjVo": bars["volume"],
    }).reset_index(drop=True)


class FakeClient:
    def __init__(self, credentials=None, **kw):
        self.credentials = credentials
        if not (credentials and credentials.api_key):
            raise jquants.JQuantsError("APIキーが見つかりません")
        self.cache_dir = Path(kw.get("cache_dir", "data/jquants"))

    def listed_info(self, on=None):
        return pd.DataFrame({"Code": ["72030", "67580"], "CoName": ["A", "B"], "Mkt": ["0111", "0111"]})

    def coverage(self, *, probe_code="7203"):
        return ("2016-01-04", None)

    def daily_quotes(self, *, code=None, on=None, start=None, end=None, clamp=True):
        return _jq_frame()

    def cached_daily_quotes(self, code, *, start, end, refresh=False):
        df = _jq_frame()
        path = self.cache_dir / "daily" / f"{code}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    def topix(self, *, start=None, end=None):
        raise jquants.JQuantsError("TOPIX の取得が失敗しました（プランにより不可）")

    def earnings_dates(self, *, start=None, end=None):
        return pd.DataFrame({"PubDate": ["2026-09-10"], "SchDate": ["2026-09-11"], "Code": ["72030"]})


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
    assert "JQUANTS_API_KEY" in html and "接続を確認" in html


def test_empty_submission_is_rejected(sandbox):
    url, root = sandbox
    r = _post(url + "/start", {"api_key": ""})
    assert r.status == 200 and "APIキーを入力してください" in r.read().decode()
    assert not (root / ".env").exists()


def test_full_pipeline_with_fake_client(sandbox):
    url, root = sandbox
    r = _post(url + "/start", {"api_key": "TESTKEY123"})
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
    assert "JQUANTS_API_KEY=TESTKEY123" in env
    if os.name != "nt":
        assert oct(os.stat(root / ".env").st_mode & 0o777) == "0o600"

    # レポートに U-1 / U-3 / 基準線が入り、認証情報は入っていない
    report = json.loads((root / "data" / "setup_report.json").read_text())
    assert report["U-1_earliest_date"] == "2016-01-04"
    assert report["U-1_coverage_start"] == "2016-01-04"
    assert report["U-3_topix_available"] is False and "1306" in report["index_used"]
    assert report["U-5_earnings_date_available"] is True
    assert "slippage_0.0" in report["baseline"] and "slippage_0.1" in report["baseline"]
    assert "TESTKEY123" not in json.dumps(report)
    assert "TESTKEY123" not in "\n".join(status["log"])
    assert "完了" in status["result_html"]


def test_to_bars_matches_official_v2_schema():
    """公式クライアントが返す列名（EQ_BARS_DAILY_COLUMNS_V2）を to_bars が読めること。

    ライブラリ側の仕様変更をここで検知する。V1 の列名を前提にしていたのが
    403 の遠因だったので、スキーマは推測せず公式定義に突き合わせる。
    """
    from jquantsapi import constants
    from short_trade.jquants import to_bars

    official = set(constants.EQ_BARS_DAILY_COLUMNS_V2)
    used = {"Date", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "O", "H", "L", "C", "Vo"}
    assert used <= official, f"公式の列定義に無い列を使っている: {used - official}"

    bars = to_bars(_jq_frame(n=30))
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 30 and bars.index.is_monotonic_increasing


def test_to_bars_falls_back_to_unadjusted_columns():
    from short_trade.jquants import to_bars
    raw = _jq_frame(n=10).drop(columns=["AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"])
    assert len(to_bars(raw)) == 10


def test_to_bars_reports_actual_columns_on_schema_change():
    from short_trade.jquants import JQuantsError, to_bars
    raw = _jq_frame(n=5).rename(columns={"AdjC": "Close", "C": "Cx"})
    with pytest.raises(JQuantsError, match="実際の列"):
        to_bars(raw)


def test_to_earnings_map_normalises_five_digit_codes():
    from short_trade.jquants import to_earnings_map
    raw = pd.DataFrame({"PubDate": ["2026-09-01"], "SchDate": ["2026-09-11"], "Code": ["72030"]})
    m = to_earnings_map(raw)
    assert list(m) == ["7203"] and m["7203"][0] == pd.Timestamp("2026-09-11")


# ---- 契約範囲の自動検出（docs/19 U-1）----
_COVERAGE_MSG = (
    "株価の取得 が失敗しました: HTTPError: 400 for url: "
    "https://api.jquants.com/v2/equities/bars/daily?code=7203&from=2008-01-01 "
    "body: Your subscription covers the following dates: 2016-09-10 ~ . "
    "If you want more data, please check other plans:https://jpx-jquants.com/#dataset"
)


def test_parse_coverage_reads_the_real_error_message():
    from short_trade.jquants import parse_coverage
    assert parse_coverage(_COVERAGE_MSG) == ("2016-09-10", None)
    assert parse_coverage("subscription covers the following dates: 2021-01-04 ~ 2026-06-18 .") == (
        "2021-01-04", "2026-06-18")
    assert parse_coverage("まったく別のエラー") is None


def test_daily_quotes_clamps_to_subscription_range():
    """契約範囲外を要求したら、範囲内へ丸めて自動で取り直す。"""
    from short_trade.jquants import Credentials, JQuantsClient

    calls = []

    class Inner:
        def get_eq_bars_daily(self, code="", from_yyyymmdd="", to_yyyymmdd="", date_yyyymmdd=""):
            calls.append(from_yyyymmdd)
            if from_yyyymmdd and from_yyyymmdd < "2016-09-10":
                raise RuntimeError(_COVERAGE_MSG)
            return _jq_frame(n=5)

    c = JQuantsClient(Credentials(api_key="k"))
    c._client = Inner()
    df = c.daily_quotes(code="7203", start="2008-01-01")
    assert len(df) == 5
    assert calls == ["2008-01-01", "2016-09-10"], calls
    assert c._coverage == ("2016-09-10", None)


def test_coverage_is_probed_once_and_cached():
    from short_trade.jquants import Credentials, JQuantsClient

    calls = []

    class Inner:
        def get_eq_bars_daily(self, code="", from_yyyymmdd="", to_yyyymmdd="", date_yyyymmdd=""):
            calls.append(from_yyyymmdd)
            if from_yyyymmdd < "2016-09-10":
                raise RuntimeError(_COVERAGE_MSG)
            return _jq_frame(n=3)

    c = JQuantsClient(Credentials(api_key="k"))
    c._client = Inner()
    assert c.coverage() == ("2016-09-10", None)
    assert c.coverage() == ("2016-09-10", None)
    assert calls == ["1990-01-01"], "2回目は問い合わせ直さないこと"


def test_unrelated_error_is_not_swallowed_by_clamp():
    from short_trade.jquants import Credentials, JQuantsClient, JQuantsError

    class Inner:
        def get_eq_bars_daily(self, **kw):
            raise RuntimeError("500 Internal Server Error")

    c = JQuantsClient(Credentials(api_key="k"))
    c._client = Inner()
    with pytest.raises(JQuantsError, match="500"):
        c.daily_quotes(code="7203", start="2008-01-01")
