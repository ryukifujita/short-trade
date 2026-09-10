#!/bin/bash
# short-trade 2回目以降（macOS 用）
# 決算日データが無ければ取得し、リスク率×スリッページの比較表を出します。setup 済みが前提です。

cd "$(dirname "$0")" || exit 1
echo "==============================================="
echo " short-trade 比較レポート"
echo "==============================================="
echo ""
if [ ! -f ".env" ] || [ ! -d ".venv" ]; then
  echo "【先に start_mac.command を実行してください】初回セットアップがまだです。"
  read -r -p "Enter キーで閉じます..."; exit 1
fi
export PYTHONPATH=src
if [ ! -f "data/jquants/earnings.parquet" ]; then
  echo "決算発表予定日を取得します（初回のみ。数分〜十数分かかります）..."
  ./.venv/bin/python -m short_trade fetch --earnings || echo "（決算日の取得に失敗しました。決算跨ぎ禁止は無効のまま続けます）"
  echo ""
fi
./.venv/bin/python -m short_trade compare
echo ""
read -r -p "Enter キーで閉じます..."
