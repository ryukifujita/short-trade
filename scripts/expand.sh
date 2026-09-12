#!/bin/bash
# ユニバース拡張（1回だけ）。expand_mac.command から呼ばれる。
#   最新化 → 環境確認 → プライム時価総額上位 300 銘柄の日足を取得 → 決算日を追加取得
cd "$(dirname "$0")/.." || exit 1
. scripts/common.sh
echo "==============================================="
echo " short-trade ユニバース拡張（300銘柄）"
echo "==============================================="
ensure_python
self_update
. scripts/common.sh
ensure_venv
if [ ! -f ".env" ]; then
  echo ""
  echo "【APIキーが未設定です】start_mac.command を先に実行してください。"
  pause_exit 1
fi
export PYTHONPATH=src
echo ""
echo "プライム市場の時価総額上位 300 銘柄の日足を取得します（数分〜十数分。1回だけ）..."
./.venv/bin/python -m short_trade fetch --universe --market プライム --top 300 || pause_exit 1
echo ""
echo "追加した銘柄の決算発表予定日を取得します..."
./.venv/bin/python -m short_trade fetch --earnings || echo "（決算日の取得に失敗。次回 run_mac.command で再試行します）"
echo ""
echo "完了。以後は run_mac.command を開けば 300 銘柄で比較が走ります（30銘柄のときより時間がかかります。目安 10〜20 分）。"
pause_exit 0
