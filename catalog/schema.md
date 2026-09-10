# 戦略仕様フォーマット（catalog/rules/*.yaml）

`docs/17` §17.5 で定義したフォーマットの正式版。FR-C05〜C07 に対応する。

## 原則

1. **1戦略＝1ファイル。** ファイル名は戦略ID（`ST-06.yaml`）。
2. **共通部分は書かない。** `catalog/common.yaml` を継承し、差分だけを書く。
3. **同じ記述からバックテストと実運用の両方が動く**こと（VR-001）。コードに条件を書かない。
4. **`params_to_optimize` に列挙した以外のパラメータを探索しない**（FR-C06）。ここが自由度の宣言であり、
   VR-050（多重検定の評価）の入力になる。
5. **原法からの改変は `deviations` に必ず書く。** 何を捨てたか、なぜか、成績にどう影響しうるかを残す。

## キー一覧

| キー | 必須 | 内容 |
|---|---|---|
| `id` / `name` / `factor` | ○ | カタログ（`strategies.yaml`）と一致させる |
| `hypothesis` | ○ | この戦略が取りにいく収益の源泉を1文で |
| `extends` | ○ | 継承元（`common.yaml`） |
| `universe` | — | 共通ユニバースへの追加条件 |
| `regime_filter` | — | この戦略固有の稼働条件（DEC-003） |
| `entry.conditions` | ○ | すべて満たしたときにシグナル。式は確定足のみを参照する |
| `entry.rank` | — | 候補が上限を超えたときの優先順位 |
| `exit.stop` | ○ | 論理ストップの価格式。**必須**（DEC-004） |
| `exit.rules` | ○ | 手仕舞い条件（トレーリング・目標・時間） |
| `sizing` | — | 共通と異なる場合のみ |
| `params_to_optimize` | ○ | 探索する自由度。name / default / range / step |
| `deviations` | ○ | 原法からの改変とその理由 |
| `expected_correlation` | ○ | 他戦略との相関の**事前予想**。検証前に書く（`docs/18` §18.4） |
| `data_required` | ○ | 必要データ。共通データ以外を明記 |

## 式の記法

- `close` / `open` / `high` / `low` / `volume` … 当日の確定足
- `close[-n]` … n営業日前
- `SMA(x, n)` / `EMA(x, n)` / `ATR(n)` / `RSI(n)` … 指標。すべて確定足のみを参照（FR-144）
- `HIGHEST(high, n)` / `LOWEST(low, n)` … 直近n本の最高値・最安値（当日を含む）
- `PCTRANK(x, n)` … 直近n本の分布における当日値のパーセンタイル（0〜100）
- `INDEX.close` … TOPIX
- 指標の実装はバックテストと実運用で共有する（VR-001）。二重実装を禁止する。
