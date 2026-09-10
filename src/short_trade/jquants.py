"""J-Quants API クライアントとローカルキャッシュ（DR-031 / DR-032）。

認証情報の渡し方（NFR-020: 平文で保存しない）:
    JQUANTS_REFRESH_TOKEN   … 推奨。リフレッシュトークンのみを渡す（有効期限は1週間）
    JQUANTS_MAIL_ADDRESS / JQUANTS_PASSWORD … 上が無いときのみ使う

    パスワードよりリフレッシュトークンを推奨するのは、漏洩時の影響を小さくするため。

注意:
    このモジュールは **実際の API に対しては未検証** である。開発環境から
    api.jquants.com への通信が遮断されていたため、疎通確認ができていない。
    最初の実行時はレスポンス形式を必ず目視で確認すること（docs/19 §19.7 の U-1〜U-5）。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

BASE = "https://api.jquants.com/v1"
# カレントディレクトリ相対にすると、別の場所から実行したときに CLI と読み書き先がずれる。
# リポジトリ直下の data/jquants に固定する。
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "jquants"


class JQuantsError(RuntimeError):
    pass


def _load_dotenv(path: Path | None = None) -> None:
    """リポジトリ直下の .env を環境変数に読み込む（既にある変数は上書きしない）。"""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Credentials:
    refresh_token: str | None = None
    mail_address: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls) -> "Credentials":
        _load_dotenv()
        c = cls(
            refresh_token=os.getenv("JQUANTS_REFRESH_TOKEN"),
            mail_address=os.getenv("JQUANTS_MAIL_ADDRESS"),
            password=os.getenv("JQUANTS_PASSWORD"),
        )
        if not c.refresh_token and not (c.mail_address and c.password):
            raise JQuantsError(
                "認証情報が見つかりません。JQUANTS_REFRESH_TOKEN を設定するか、"
                "JQUANTS_MAIL_ADDRESS と JQUANTS_PASSWORD を設定してください。"
            )
        return c


class JQuantsClient:
    def __init__(self, credentials: Credentials | None = None, *,
                 cache_dir: Path = DEFAULT_CACHE, timeout: int = 30,
                 min_interval: float = 0.2):
        self.credentials = credentials or Credentials.from_env()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_interval = min_interval          # レートリミット対策（U-4 で実測して調整）
        self._id_token: str | None = None
        self._last_call = 0.0

    # ------------------------------------------------------------ 認証
    def _refresh_token(self) -> str:
        if self.credentials.refresh_token:
            return self.credentials.refresh_token
        r = requests.post(
            f"{BASE}/token/auth_user",
            data=json.dumps({"mailaddress": self.credentials.mail_address,
                             "password": self.credentials.password}),
            timeout=self.timeout,
        )
        self._raise_for_status(r, "auth_user")
        return r.json()["refreshToken"]

    @property
    def id_token(self) -> str:
        if self._id_token:
            return self._id_token
        r = requests.post(
            f"{BASE}/token/auth_refresh",
            params={"refreshtoken": self._refresh_token()},
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise JQuantsError(
                f"auth_refresh が失敗しました: HTTP {r.status_code}。"
                "リフレッシュトークンの有効期限は1週間です。期限切れなら JQUANTS_REFRESH_TOKEN を取り直してください"
            )
        self._id_token = r.json()["idToken"]
        return self._id_token

    @staticmethod
    def _raise_for_status(r: requests.Response, what: str) -> None:
        if r.status_code >= 400:
            # 本文に認証情報が混ざらないよう、先頭のみを出す（NFR-021）
            raise JQuantsError(f"{what} が失敗しました: HTTP {r.status_code} {r.text[:200]}")

    # ------------------------------------------------------------ 取得
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        """pagination_key をたどって全ページを返す。"""
        params = dict(params or {})
        while True:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            r = requests.get(
                f"{BASE}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self.id_token}"},
                timeout=self.timeout,
            )
            self._last_call = time.monotonic()
            if r.status_code == 401:            # idToken の期限切れ。1度だけ取り直す
                self._id_token = None
                r = requests.get(
                    f"{BASE}{path}", params=params,
                    headers={"Authorization": f"Bearer {self.id_token}"},
                    timeout=self.timeout,
                )
            self._raise_for_status(r, f"GET {path}")
            payload = r.json()
            yield payload
            key = payload.get("pagination_key")
            if not key:
                return
            params["pagination_key"] = key

    def _collect(self, path: str, field: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        for page in self._get(path, params):
            rows.extend(page.get(field, []))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ 各API
    def listed_info(self, on: date | str | None = None) -> pd.DataFrame:
        """上場銘柄一覧。生存者バイアス対策のため日付を指定して保存する（VR-004）。"""
        params = {"date": _ymd(on)} if on else {}
        return self._collect("/listed/info", "info", params)

    def daily_quotes(self, *, code: str | None = None, on: date | str | None = None,
                     start: date | str | None = None, end: date | str | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if code:
            params["code"] = code
        if on:
            params["date"] = _ymd(on)
        if start:
            params["from"] = _ymd(start)
        if end:
            params["to"] = _ymd(end)
        return self._collect("/prices/daily_quotes", "daily_quotes", params)

    def topix(self, *, start=None, end=None) -> pd.DataFrame:
        """無料プランで取得できない可能性がある（U-3）。その場合は 1306 で代用する。"""
        params = {}
        if start:
            params["from"] = _ymd(start)
        if end:
            params["to"] = _ymd(end)
        return self._collect("/indices/topix", "topix", params)

    def announcement(self) -> pd.DataFrame:
        """翌営業日の決算発表予定（RM-001d / ST-23）。取得可否は U-5 で確認する。"""
        return self._collect("/fins/announcement", "announcement")

    # ------------------------------------------------------------ キャッシュ
    def cached_daily_quotes(self, code: str, *, start, end, refresh: bool = False) -> pd.DataFrame:
        path = self.cache_dir / "daily" / f"{code}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        df = self.daily_quotes(code=code, start=start, end=end)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.attrs["source"] = "jquants"
            df.attrs["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            df.to_parquet(path)
        return df


def _ymd(d: date | str | None) -> str:
    if d is None:
        return ""
    return d if isinstance(d, str) else d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 正規化

_COLUMN_MAP = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    "AdjustmentOpen": "open", "AdjustmentHigh": "high", "AdjustmentLow": "low",
    "AdjustmentClose": "close", "AdjustmentVolume": "volume",
}


def to_bars(raw: pd.DataFrame, *, adjusted: bool = True) -> pd.DataFrame:
    """J-Quants のレスポンスを内部形式（date index / open,high,low,close,volume）へ正規化する。

    FR-105 / DR-031: 既定では分割・併合を反映した調整済み株価を使う。
    """
    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    prefix = "Adjustment" if adjusted and "AdjustmentClose" in raw.columns else ""
    cols = {f"{prefix}{k}": v for k, v in
            [("Open", "open"), ("High", "high"), ("Low", "low"),
             ("Close", "close"), ("Volume", "volume")]}
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise JQuantsError(f"想定した列がありません: {missing}. 実際の列: {list(raw.columns)}")

    df = raw.rename(columns=cols)[list(cols.values())].astype(float)
    df.index = pd.to_datetime(raw["Date"])
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # DR-020: タイムスタンプの単調性を検査する
    if not df.index.is_monotonic_increasing:
        raise JQuantsError("日付が単調増加ではありません")
    return df.dropna(subset=["close"])
