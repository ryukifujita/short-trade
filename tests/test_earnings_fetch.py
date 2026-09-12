"""決算発表予定日の再開可能な取得（fetch --earnings）の検査。"""
import json
from argparse import Namespace

import pandas as pd
import pytest

from short_trade import cli
from short_trade.jquants import Credentials, JQuantsClient


class FlakyV2:
    """公式 ClientV2 の偽物。指定した日を一度だけ失敗させる。"""

    def __init__(self, fail_once: set[str]):
        self.fail_once = set(fail_once)
        self.calls: list[str] = []

    def get_fin_earnings_date(self, *, date_yyyymmdd: str):
        self.calls.append(date_yyyymmdd)
        if date_yyyymmdd in self.fail_once:
            self.fail_once.discard(date_yyyymmdd)
            raise RuntimeError("HTTP 429 Too Many Requests")
        if date_yyyymmdd.endswith("-05"):        # 5日にだけ予定が公表されたことにする
            return pd.DataFrame({"PubDate": [date_yyyymmdd], "SchDate": [date_yyyymmdd], "Code": ["72030"]})
        return pd.DataFrame()


@pytest.fixture
def client(tmp_path, monkeypatch):
    c = JQuantsClient(Credentials(api_key="dummy"), cache_dir=tmp_path / "cache")
    c._client = FlakyV2(fail_once={"2024-01-03"})
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return c


def test_by_day_retries_transient_failures_and_keeps_partial_rows(client):
    days = [f"2024-01-{d:02d}" for d in range(1, 8)]
    df, failed = client.earnings_dates_by_day(days, retries=2, workers=2)
    assert failed == []
    assert client._client.calls.count("2024-01-03") == 2       # 1回失敗 → 再試行で成功
    assert len(df) == 1 and df["Code"].iloc[0] == "72030"


def test_by_day_reports_permanent_failures_without_losing_the_rest(client):
    client._client.fail_once = {"2024-01-02"}
    orig = client._client.get_fin_earnings_date

    def always_fail(*, date_yyyymmdd):
        if date_yyyymmdd == "2024-01-02":
            raise RuntimeError("HTTP 500")
        return orig(date_yyyymmdd=date_yyyymmdd)

    client._client.get_fin_earnings_date = always_fail
    df, failed = client.earnings_dates_by_day([f"2024-01-{d:02d}" for d in range(1, 8)], retries=2)
    assert failed == ["2024-01-02"]
    assert len(df) == 1                                          # 他の日の結果は残る


def test_cli_fetch_earnings_is_resumable(tmp_path, monkeypatch, client, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "DATA", tmp_path / "data" / "jquants")
    monkeypatch.setattr(cli, "EARNINGS_PATH", tmp_path / "data" / "jquants" / "earnings.parquet")
    monkeypatch.setattr(cli, "EARNINGS_PROGRESS", tmp_path / "data" / "jquants" / "earnings_progress.json")

    assert cli._fetch_earnings(client, "2024-01-01", "2024-02-10") == 0
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert prog["done_through"] == "2024-02-10" and prog["failed_days"] == []
    saved = pd.read_parquet(cli.EARNINGS_PATH)
    assert len(saved) == 2                                       # 1/5 と 2/5
    n_calls = len(client._client.calls)

    # 2回目: 続きの日だけ問い合わせる
    assert cli._fetch_earnings(client, "2024-01-01", "2024-02-15") == 0
    assert len(client._client.calls) - n_calls == 5              # 2/11〜2/15
    assert json.loads(cli.EARNINGS_PROGRESS.read_text())["done_through"] == "2024-02-15"
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 2          # 重複なし

    # 3回目: 取るものが無ければ即終了
    n_calls = len(client._client.calls)
    assert cli._fetch_earnings(client, "2024-01-01", "2024-02-15") == 0
    assert len(client._client.calls) == n_calls
    assert "取得済み" in capsys.readouterr().out


def test_cli_fetch_earnings_records_failed_days_and_retries_them_next_time(tmp_path, monkeypatch, client):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "DATA", tmp_path / "data" / "jquants")
    monkeypatch.setattr(cli, "EARNINGS_PATH", tmp_path / "data" / "jquants" / "earnings.parquet")
    monkeypatch.setattr(cli, "EARNINGS_PROGRESS", tmp_path / "data" / "jquants" / "earnings_progress.json")
    orig = client._client.get_fin_earnings_date
    broken = {"2024-01-05"}

    def sometimes(*, date_yyyymmdd):
        if date_yyyymmdd in broken:
            raise RuntimeError("HTTP 500")
        return orig(date_yyyymmdd=date_yyyymmdd)

    client._client.get_fin_earnings_date = sometimes
    cli._fetch_earnings(client, "2024-01-01", "2024-01-10")
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert prog["failed_days"] == ["2024-01-05"] and prog["done_through"] == "2024-01-10"
    assert not cli.EARNINGS_PATH.exists() or len(pd.read_parquet(cli.EARNINGS_PATH)) == 0

    broken.clear()                                               # 復旧後にもう一度
    cli._fetch_earnings(client, "2024-01-01", "2024-01-10")
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert prog["failed_days"] == []
    assert len(pd.read_parquet(cli.EARNINGS_PATH)) == 1


def test_cli_fetch_earnings_keeps_unattempted_holes_when_aborting(tmp_path, monkeypatch, client):
    """先の月が丸ごと失敗して打ち切っても、進捗ファイルにあった別の穴を失わない。"""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "DATA", tmp_path / "data" / "jquants")
    monkeypatch.setattr(cli, "EARNINGS_PATH", tmp_path / "data" / "jquants" / "earnings.parquet")
    monkeypatch.setattr(cli, "EARNINGS_PROGRESS", tmp_path / "data" / "jquants" / "earnings_progress.json")
    cli.EARNINGS_PROGRESS.parent.mkdir(parents=True)
    cli.EARNINGS_PROGRESS.write_text(json.dumps({"done_through": "2024-03-31", "failed_days": ["2024-03-20"]}))

    def all_fail(*, date_yyyymmdd):
        raise RuntimeError("HTTP 503")

    client._client.get_fin_earnings_date = all_fail
    cli._fetch_earnings(client, "2024-01-01", "2024-04-05")
    prog = json.loads(cli.EARNINGS_PROGRESS.read_text())
    assert "2024-03-20" in prog["failed_days"]                    # 最初の塊（3月の穴）は試して失敗
    assert prog["done_through"] == "2024-03-31"                   # 4月には進んでいない
    # 4月は未着手なので failed_days には入らず、done_through から再生成される
    assert all(d.startswith("2024-03") for d in prog["failed_days"])
