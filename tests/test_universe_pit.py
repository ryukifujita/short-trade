"""時点ユニバース（VR-004）の検査: 所属フラグ、市場区分の選別、取得コマンド。"""
import json
from argparse import Namespace

import pandas as pd
import pytest

from short_trade import cli
from short_trade.backtest import BacktestConfig, membership_flag, run, universe_filters
from short_trade.jquants import Credentials, JQuantsClient
from short_trade.spec import load_all
from synthetic import flat_then_breakout, rising_index

CATALOG = cli.CATALOG


def test_membership_flag_is_valid_from_snapshot_until_next_snapshot():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    m = {pd.Timestamp("2024-01-03"): {"A"}, pd.Timestamp("2024-01-07"): {"B"}}
    a = membership_flag(m, "A", idx)
    b = membership_flag(m, "B", idx)
    assert a.tolist() == [False, False, True, True, True, True, False, False, False, False]
    assert b.tolist() == [False] * 6 + [True] * 4
    assert not membership_flag(m, "C", idx).any()                # どの基準日にも無い銘柄は常に対象外


def test_membership_blocks_entries_outside_the_window():
    spec = next(s for s in load_all(CATALOG) if s.id == "ST-06")
    df = flat_then_breakout()
    idx = rising_index(len(df), start=str(df.index[0].date()))
    base = dict(initial_equity=1_000_000, slippage_pct=0, hysteresis_days=1, max_position_pct=100)
    with_member = run(spec, {"T": df}, index=idx,
                      config=BacktestConfig(**base, universe_membership={df.index[0]: {"T"}}))
    without = run(spec, {"T": df}, index=idx,
                  config=BacktestConfig(**base, universe_membership={df.index[0]: {"X"}}))
    assert with_member.trades and not without.trades
    names = [t for t, _ in universe_filters(BacktestConfig(**base, universe_membership={df.index[0]: {"T"}}), df, "T")]
    assert any("時点ユニバース" in n for n in names)


def test_select_prime_like_accepts_pre_2022_first_section_name():
    snap = pd.DataFrame({"Code": ["1", "2", "3"], "MktCap": [3, 2, 1],
                         "MktNm": ["プライム", "市場第一部", "グロース"]})
    got = cli._select_prime_like(snap, "プライム")
    assert got["Code"].tolist() == ["1", "2"]
    assert len(cli._select_prime_like(snap, "")) == 3
    assert cli._select_prime_like(snap, "グロース")["Code"].tolist() == ["3"]


class FakeV2:
    """基準日ごとに上位銘柄が入れ替わる偽クライアント。"""

    def __init__(self):
        self.bars_calls: list[str] = []

    def get_eq_bars_daily(self, *, code="", from_yyyymmdd="", to_yyyymmdd="", date_yyyymmdd=""):
        if date_yyyymmdd:                                        # スナップショット
            if date_yyyymmdd.endswith("-01"):                    # 1日は休日ということにする
                return pd.DataFrame()
            year = int(date_yyyymmdd[:4])
            codes = ["10000", "20000"] if year < 2020 else ["20000", "30000"]
            return pd.DataFrame({"Code": codes, "MktCap": [2.0, 1.0], "Date": [date_yyyymmdd] * 2})
        self.bars_calls.append(code)
        return pd.DataFrame({"Date": ["2018-01-04"], "Code": [code], "O": [1.0], "H": [1.0], "L": [1.0], "C": [1.0], "Vo": [1.0]})

    def get_list(self, *, date_yyyymmdd=""):
        return pd.DataFrame({"Code": ["10000", "20000", "30000"], "CoName": ["A", "B", "C"],
                             "MktNm": ["市場第一部", "市場第一部", "プライム"]})


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data" / "jquants"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "DATA", data)
    monkeypatch.setattr(cli, "UNIVERSE_PIT_PATH", data / "universe_pit.json")
    client = JQuantsClient(Credentials(api_key="dummy"), cache_dir=data)
    client._client = FakeV2()
    return client, data


def test_fetch_universe_pit_builds_snapshots_and_union(env):
    client, data = env
    args = Namespace(every=12, market="プライム", top=2)
    assert cli._fetch_universe_pit(client, "2018-01-01", args) == 0
    saved = json.loads(cli.UNIVERSE_PIT_PATH.read_text(encoding="utf-8"))
    members = saved["members"]
    assert all(not d.endswith("-01") for d in members)           # 休日は翌営業日にずらす
    early = [d for d in members if d < "2020"]; late = [d for d in members if d >= "2020"]
    assert early and late
    assert members[early[0]] == ["1000", "2000"] and members[late[0]] == ["2000", "3000"]   # 4桁に正規化
    assert sorted(set(client._client.bars_calls)) == ["1000", "2000", "3000"]              # 和集合を取得
    m = cli._load_membership()
    assert m and all(isinstance(k, pd.Timestamp) for k in m)
    assert {"1000", "2000"} in m.values() and {"2000", "3000"} in m.values()
