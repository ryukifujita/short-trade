#!/bin/bash
# 2回目以降。run_mac.command から呼ばれる。
#   最新化 → 環境確認 → 決算日（無ければ取得）→ 比較表 → 相関
cd "$(dirname "$0")/.." || exit 1
. scripts/common.sh
echo "==============================================="
echo " short-trade 比較レポート"
echo "==============================================="
ensure_python               # 更新の展開に Python を使うので先に確認する
self_update
. scripts/common.sh          # 更新後の定義を読み直す
ensure_venv
if [ ! -f ".env" ]; then
  echo ""
  echo "【APIキーが未設定です】start_mac.command を先に実行してください。"
  echo "  （前のフォルダで設定済みなら、そのフォルダの .env をこのフォルダにコピーしても動きます）"
  pause_exit 1
fi
export PYTHONPATH=src
if [ ! -d "data/jquants/daily" ] || [ -z "$(ls -A data/jquants/daily 2>/dev/null)" ]; then
  echo ""
  echo "株価データがありません。まず取得します（30銘柄・数分）..."
  ./.venv/bin/python -m short_trade fetch --index topix || ./.venv/bin/python -m short_trade fetch --index 1306
  ./.venv/bin/python -m short_trade fetch --codes 7203,6758,9432,8058,4063,6098,6501,8035,9984,7974,6861,8306,4502,6902,7741,4568,6367,9433,8031,6594,4519,6954,7267,8766,6857,4661,9983,2914,8001,3382
fi
if [ ! -f "data/jquants/earnings.parquet" ]; then
  echo ""
  echo "決算発表予定日を取得します（初回のみ。数分〜十数分）..."
  ./.venv/bin/python -m short_trade fetch --earnings || echo "（決算日の取得に失敗。決算跨ぎ禁止は無効のまま続けます）"
fi
echo ""
./.venv/bin/python -m short_trade compare
echo ""
echo "----- 戦略間の相関（Phase 2 の4本） -----"
./.venv/bin/python -m short_trade correlate
pause_exit 0
