"""決算発表予定日の取得（fetch --earnings、銘柄ごと・差分更新）の検査。"""
import json
from argparse import Namespace

import pandas as pd
import pytest

from short_trade import cli
from short_trade.jquants import Credentials, JQuantsClient, to_earnings_map


class FakeV2:
    """公式 ClientV2 の偽物。code 指定で履歴つきの全レコードを返す。"""

    def __init__(self, fail_once: set[str] = (), always_fail: set[str] = ()):
        self.fail_once = set(fail_once)
        self.always_fail = set(always_fail)
        self.calls: list[str] = []

    def get_fin_earnings_date(self, *, code: str = "", date_yyyymmdd: str = "", scheduled_date_yyyymmdd: str = ""):
        self.calls.append(code or date_yyyymmdd)
        if code in self.always_fail:
            raise RuntimeError("HTTP 500")
        if code in self.fail_once:
            self.fail_once.discard(code)
            raise RuntimeError("HTTP 429 Too Many Requests")
        if code:
            c5 = code + "0"
            return pd.DataFrame({
                "PubDate": ["2024-01-10", "2024-04-01", "2024-04-20"],
                "SchDate": ["2024-02-05", "2024-05-10", "未定"],       # 変更履歴と「未定」
                "FYE": ["2024-03", "2024-03", "2024-03"], "FQName": ["3Q", "FY", "FY"],
                "Code": [c5, c5, c5], "CoName": ["X", "X", "X"], "CoNameEn": ["X", "X", "X"],
            })
        return pd.DataFrame()


@pytest.fixture
def client(tmp_path, monkeypatch):
    c = JQuantsClient(Credentials(api_key="dummy"), cache_dir=tmp_path / "cache")
    c._client = FakeV2(fail_once={"6758"})
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return c


@pytest.fixture
def paths(tmp_path, monkeypatch):
    data = tmp_path / "data" / "jquants"
    (data / "daily").mkdir(parents=True)
    for code in ("7203", "6758", "9432"):
        pd.DataFrame({"Date": ["2024-01-04"], "C": [1.0]}).to_parquet(data / "daily" / f"{code}.parquet")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "DATA", data)
    monkeypatch.setattr(cli, "EARNINGS_PATH", data / "earnings.parquet")
    monkeypatch.setattr(cli, "EARNINGS_PROGRESS", data / "earnings_progress.json")
    return data


def test_for_codes_retries_and_returns_all_history(client):
    df, failed = client.earnings_dates_for_codes(["7203", "6758"], retries=2, workers=2)
    assert failed == []
    assert client._client.calls.count("6758") == 2               # 1回失敗 → 再試行で成功
    assert len(df) == 6 and set(df["Code"]) == {"72030", "67580"}


def test_for_codes_reports_permanent_failures_without_losing_the_rest(client):
    client._client.always_fail = {"9432"}
    df, failed = client.earnings_dates_for_codes(["7203", "9432"], retries=2)
    assert failed == ["9432"] and set(df["Code"]) == {"72030"}


def test_to_earnings_map_uses_schdate_drops_undecided_and_keeps_history(client):
    df, _ = client.earnings_dates_for_codes(["7203"])
    m = to_earnings_map(df)
    assert list(m) == ["7203"]                                    # 5桁 → 4桁
    assert m["7203"] == [pd.Timestamp("2024-02-05"), pd.Timestamp("2024-05-10")]   # 「未定」は捨てる


def test_cli_fetch_earnings_fetches_cached_symbols_and_is_incremental(paths, client, capsys):
    assert cli._fetch_earnings(client, "2016-09-12", None) == 0
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert set(prog["fetched"]) == {"7203", "6758", "9432"} and prog["failed"] == []
    saved = pd.read_parquet(cli.EARNINGS_PATH)
    assert len(saved) == 9
    out = capsys.readouterr().out
    assert "3/3 銘柄に日付あり" in out

    n = len(client._client.calls)
    assert cli._fetch_earnings(client, "2016-09-12", None) == 0   # 新しいので取り直さない
    assert len(client._client.calls) == n
    assert "取得済み" in capsys.readouterr().out

    # 銘柄が増えたらその銘柄だけ取る
    pd.DataFrame({"Date": ["2024-01-04"], "C": [1.0]}).to_parquet(paths / "daily" / "8058.parquet")
    cli._fetch_earnings(client, "2016-09-12", None)
    assert client._client.calls[n:] == ["8058"]
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 12


def test_cli_fetch_earnings_retries_failed_symbols_next_time(paths, client):
    client._client.always_fail = {"9432"}
    cli._fetch_earnings(client, "2016-09-12", None)
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert prog["failed"] == ["9432"] and "9432" not in prog["fetched"]
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 6

    client._client.always_fail = set()
    n = len(client._client.calls)
    cli._fetch_earnings(client, "2016-09-12", None)
    assert client._client.calls[n:] == ["9432"]
    assert json.loads(cli.EARNINGS_PROGRESS.read_text())["failed"] == []
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 9


def test_stale_symbols_are_refreshed_after_max_age(paths, client):
    cli._fetch_earnings(client, "2016-09-12", None)
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    prog["fetched"]["7203"] = "2000-01-01T00:00:00"
    cli.EARNINGS_PROGRESS.write_text(json.dumps(prog))
    n = len(client._client.calls)
    cli._fetch_earnings(client, "2016-09-12", None)
    assert client._client.calls[n:] == ["7203"]
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 9           # 取り直しても重複しない


def test_compare_report_counts_symbols_with_earnings(paths, client, monkeypatch):
    """決算日が1銘柄にも一致しなければ、レポートにそれが出る（docs/27 §27.1 の再発防止）。"""
    import numpy as np
    from synthetic import make_bars, rising_index
    for f in (paths / "daily").glob("*.parquet"):
        f.unlink()
    n = 400
    for i, code in enumerate(("7203", "6758")):
        bars = make_bars([1000 * (1.002 + i * 0.0005) ** k for k in range(n)])
        pd.DataFrame({"Date": bars.index.strftime("%Y-%m-%d"), "Code": "00000",
                      "O": bars["open"], "H": bars["high"], "L": bars["low"], "C": bars["close"], "Vo": bars["volume"]}
                     ).reset_index(drop=True).to_parquet(paths / "daily" / f"{code}.parquet")
    rising_index(n, start="2024-01-01").to_parquet(paths / "index.parquet")
    (cli.ROOT / "data").mkdir(exist_ok=True)
    # 一致しない銘柄コードの決算日
    pd.DataFrame({"PubDate": ["2024-06-01"], "SchDate": ["2024-06-10"], "Code": ["99990"]}).to_parquet(cli.EARNINGS_PATH)
    cli.cmd_compare(Namespace(strategy="ST-06", phase=None, risk="1.0", slippage_levels="0.0", equity=1_000_000, start=None, end=None))
    rep = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())
    assert rep["earnings_applied"] is True and rep["earnings_symbols_matched"] == 0
    # 一致する銘柄なら数えられる
    pd.DataFrame({"PubDate": ["2024-06-01"], "SchDate": ["2024-06-10"], "Code": ["72030"]}).to_parquet(cli.EARNINGS_PATH)
    cli.cmd_compare(Namespace(strategy="ST-06", phase=None, risk="1.0", slippage_levels="0.0", equity=1_000_000, start=None, end=None))
    rep = json.loads((cli.ROOT / "data" / "compare_report.json").read_text())
    assert rep["earnings_symbols_matched"] == 1
