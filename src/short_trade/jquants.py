"""J-Quants API v2 クライアントとローカルキャッシュ（DR-031 / DR-032）。

**v1 の認証は廃止された。** 旧版の `/v1/token/auth_user`・`/v1/token/auth_refresh`
（メール／パスワード → リフレッシュトークン → IDトークン）は 2026年6月1日に終了し、
v2 ではダッシュボードで発行した **APIキーを `x-api-key` ヘッダに載せる**方式になった。
経緯は docs/23 を参照。

エンドポイントとフィールド名の推測を避けるため、**JPX 公式の Python クライアント
（`jquants-api-client` の `ClientV2`）を土台にする。** ここはその薄いラッパで、
役割は「内部形式への正規化」と「ローカルキャッシュ」に限る。

認証情報の渡し方（NFR-020: 平文で保存しない）:
    JQUANTS_API_KEY   … ダッシュボードで発行したAPIキー
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "jquants"


class JQuantsError(RuntimeError):
    pass


# 契約範囲外の日付を要求したときに API が返すメッセージ。
#   "Your subscription covers the following dates: 2016-09-10 ~ ."
# 契約プランごとに遡れる期間が違うため、範囲は決め打ちせずここから読み取る（docs/19 U-1）。
_COVERAGE_RE = re.compile(
    r"subscription covers the following dates:\s*"
    r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})?"
)


def parse_coverage(message: str) -> tuple[str, str | None] | None:
    """エラーメッセージから契約が覆う日付範囲を取り出す。読めなければ None。"""
    m = _COVERAGE_RE.search(message)
    return (m.group(1), m.group(2)) if m else None


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
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Credentials":
        _load_dotenv()
        key = os.getenv("JQUANTS_API_KEY")
        if not key:
            raise JQuantsError(
                "APIキーが見つかりません。JQUANTS_API_KEY を設定するか、"
                "`python -m short_trade setup` の画面から入力してください。"
            )
        return cls(api_key=key)


class JQuantsClient:
    """`ClientV2` の薄いラッパ。取得と正規化とキャッシュだけを受け持つ。"""

    def __init__(self, credentials: Credentials | None = None, *,
                 cache_dir: Path = DEFAULT_CACHE):
        self.credentials = credentials or Credentials.from_env()
        if not self.credentials.api_key:
            raise JQuantsError("APIキーが空です")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any | None = None
        self._coverage: tuple[str, str | None] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import jquantsapi
            except ImportError as e:  # pragma: no cover
                raise JQuantsError(
                    "jquants-api-client が入っていません。"
                    "`pip install -r requirements.txt` を実行してください。"
                ) from e
            self._client = jquantsapi.ClientV2(api_key=self.credentials.api_key)
        return self._client

    def _call(self, what: str, fn, *args, **kwargs) -> pd.DataFrame:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise JQuantsError(f"{what} が失敗しました: {type(e).__name__}: {str(e)[:300]}") from e

    # ------------------------------------------------------------ 各API
    def listed_info(self, on: date | str | None = None) -> pd.DataFrame:
        """上場銘柄一覧（v2: /equities/master）。

        生存者バイアス対策（VR-004）のため、日付を指定して各時点の状態を保存する。
        """
        return self._call("上場銘柄一覧の取得", self.client.get_list, date_yyyymmdd=_ymd(on))

    def coverage(self, *, probe_code: str = "7203") -> tuple[str, str | None]:
        """契約が覆う日付範囲 (開始日, 終了日) を確定する。

        プランごとに遡れる期間が違い、事前に知る方法がない。
        わざと古い日付を要求し、返ってくるエラーメッセージから読み取る。
        範囲内だった場合はその日付を開始日として扱う。
        """
        if self._coverage is not None:
            return self._coverage
        probe = "1990-01-01"
        try:
            self.daily_quotes(code=probe_code, start=probe, end="1990-01-31", clamp=False)
            self._coverage = (probe, None)
        except JQuantsError as e:
            found = parse_coverage(str(e))
            if found is None:
                raise
            self._coverage = found
        return self._coverage

    def daily_quotes(self, *, code: str | None = None, on: date | str | None = None,
                     start: date | str | None = None, end: date | str | None = None,
                     clamp: bool = True) -> pd.DataFrame:
        """株価四本値（v2: /equities/bars/daily）。

        clamp=True なら、契約範囲外の日付を要求したときに範囲内へ丸めて取り直す。
        """
        try:
            return self._call(
                "株価の取得", self.client.get_eq_bars_daily,
                code=code or "", from_yyyymmdd=_ymd(start), to_yyyymmdd=_ymd(end),
                date_yyyymmdd=_ymd(on),
            )
        except JQuantsError as e:
            found = parse_coverage(str(e)) if clamp else None
            if found is None:
                raise
            self._coverage = found
            cov_start, cov_end = found
            return self._call(
                "株価の取得（契約範囲に丸めて再試行）", self.client.get_eq_bars_daily,
                code=code or "", from_yyyymmdd=cov_start,
                to_yyyymmdd=_ymd(end) or (cov_end or ""), date_yyyymmdd=_ymd(on),
            )

    def topix(self, *, start=None, end=None) -> pd.DataFrame:
        """TOPIX 四本値（v2: /indices/bars/daily/topix）。

        契約プランによっては取得できない。その場合は 1306（TOPIX連動ETF）で代用する
        （docs/19 §19.4）。
        """
        return self._call("TOPIX の取得", self.client.get_idx_bars_daily_topix,
                          from_yyyymmdd=_ymd(start), to_yyyymmdd=_ymd(end))

    def earnings_dates(self, *, start=None, end=None) -> pd.DataFrame:
        """決算発表予定日（v2: /fins/earnings-date）。RM-001d / ST-23 で使う。

        公式クライアントの範囲取得は、期間内の **1日でも失敗すると全体が例外で落ち、
        取れた分も捨てられる**。長期間（約3,650日）を取るときは `earnings_dates_by_day` を使う。
        """
        kwargs = {}
        if start:
            kwargs["start_dt"] = _ymd(start)
        if end:
            kwargs["end_dt"] = _ymd(end)
        return self._call("決算発表予定日の取得", self.client.get_fin_earnings_date_range, **kwargs)

    def earnings_dates_by_day(self, days: list[str], *, retries: int = 3, workers: int = 4,
                              progress=None) -> tuple[pd.DataFrame, list[str]]:
        """決算発表予定日を日ごとに取り、失敗した日は再試行し、それでも駄目な日は名前を挙げて返す。

        返り値: (取れた行の DataFrame, 取れなかった日のリスト)。1日の失敗で全体を捨てない。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def one(day: str) -> pd.DataFrame:
            last: Exception | None = None
            for attempt in range(retries):
                try:
                    return self.client.get_fin_earnings_date(date_yyyymmdd=day)
                except Exception as e:      # 429/5xx/一時的な切断
                    last = e
                    time.sleep(1.0 * (2 ** attempt))
            raise JQuantsError(f"{day}: {type(last).__name__}: {str(last)[:120]}")

        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(one, d): d for d in days}
            for fut in as_completed(futures):
                day = futures[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        frames.append(df)
                except JQuantsError as e:
                    failed.append(day)
                    if progress:
                        progress(f"  取得失敗（後で再試行します）: {e}")
                done += 1
                if progress and (done % 200 == 0 or done == len(days)):
                    progress(f"  [{done}/{len(days)} 日] 取得済み {sum(len(f) for f in frames):,} 件")
        out = pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()
        return out, sorted(failed)

    # ------------------------------------------------------------ キャッシュ
    def cached_daily_quotes(self, code: str, *, start, end=None,
                            refresh: bool = False) -> pd.DataFrame:
        path = self.cache_dir / "daily" / f"{code}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        df = self.daily_quotes(code=code, start=start, end=end)
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.attrs["source"] = "jquants-v2"
            df.attrs["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            df.to_parquet(path)
        return df


def _ymd(d: date | str | None) -> str:
    if d is None:
        return ""
    return d if isinstance(d, str) else d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 正規化

# v2 のフィールド名。調整済み（Adj*）を優先し、無ければ素の値を使う（FR-105）。
_ADJUSTED = {"AdjO": "open", "AdjH": "high", "AdjL": "low", "AdjC": "close", "AdjVo": "volume"}
_RAW = {"O": "open", "H": "high", "L": "low", "C": "close", "Vo": "volume"}


def to_bars(raw: pd.DataFrame, *, adjusted: bool = True) -> pd.DataFrame:
    """J-Quants v2 のレスポンスを内部形式（date index / open,high,low,close,volume）へ正規化する。"""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if raw is None or raw.empty:
        return empty

    mapping = _ADJUSTED if (adjusted and set(_ADJUSTED) <= set(raw.columns)) else _RAW
    missing = [c for c in mapping if c not in raw.columns]
    if missing:
        raise JQuantsError(
            f"想定した列がありません: {missing}。実際の列: {list(raw.columns)}。"
            "J-Quants の仕様変更の可能性があります"
        )
    if "Date" not in raw.columns:
        raise JQuantsError(f"Date 列がありません。実際の列: {list(raw.columns)}")

    df = raw.rename(columns=mapping)[list(mapping.values())].astype(float)
    df.index = pd.to_datetime(raw["Date"])
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if not df.index.is_monotonic_increasing:                       # DR-020
        raise JQuantsError("日付が単調増加ではありません")
    return df.dropna(subset=["close"])


def to_index(raw: pd.DataFrame) -> pd.DataFrame:
    """TOPIX / 指数のレスポンスを `close` 列だけの DataFrame へ正規化する。"""
    if raw is None or raw.empty:
        raise JQuantsError("指数データが空です")
    if "C" not in raw.columns or "Date" not in raw.columns:
        raise JQuantsError(f"想定した列がありません。実際の列: {list(raw.columns)}")
    df = pd.DataFrame({"close": raw["C"].astype(float)})
    df.index = pd.to_datetime(raw["Date"])
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def to_earnings_map(raw: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """決算発表予定日を {銘柄コード: [日付, ...]} へ。BacktestConfig.earnings_dates に渡す。"""
    if raw is None or raw.empty:
        return {}
    col = "SchDate" if "SchDate" in raw.columns else "PubDate"
    if col not in raw.columns or "Code" not in raw.columns:
        raise JQuantsError(f"想定した列がありません。実際の列: {list(raw.columns)}")
    out: dict[str, list[pd.Timestamp]] = {}
    for code, group in raw.groupby("Code"):
        # 内部では4桁コードで扱う。v2 は5桁（末尾0）で返すことがある
        key = str(code)[:4] if len(str(code)) == 5 and str(code).endswith("0") else str(code)
        out.setdefault(key, []).extend(pd.to_datetime(group[col]).tolist())
    return out
