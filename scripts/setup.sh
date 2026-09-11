#!/bin/bash
# 初回セットアップ。start_mac.command から呼ばれる。
cd "$(dirname "$0")/.." || exit 1
. scripts/common.sh
echo "==============================================="
echo " short-trade 初回セットアップ"
echo "==============================================="
ensure_python
ensure_venv
echo ""
echo "設定画面をブラウザに開きます。（この黒い画面は開いたままにしてください）"
echo ""
PYTHONPATH=src ./.venv/bin/python -m short_trade setup
pause_exit 0
