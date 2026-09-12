#!/bin/bash
# パラメータ探索（学習／検証の分割つき）。tune_mac.command から呼ばれる。時間がかかる（30〜60分）。
cd "$(dirname "$0")/.." || exit 1
. scripts/common.sh
echo "==============================================="
echo " short-trade パラメータ探索（学習 〜2021-12-31 / 検証 2022 以降）"
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
echo "宣言済みのパラメータ（各 3 水準）を学習期間で探索し、検証期間で既定値と比べます（30〜60 分）..."
./.venv/bin/python -m short_trade optimize --train-end 2021-12-31
pause_exit 0
