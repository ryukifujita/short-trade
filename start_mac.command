#!/bin/bash
# short-trade セットアップ（macOS 用）
# このファイルをダブルクリックすると、必要な準備をして設定画面をブラウザに開きます。

cd "$(dirname "$0")" || exit 1
echo "==============================================="
echo " short-trade セットアップ"
echo "==============================================="
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "【エラー】Python が見つかりませんでした。"
  echo ""
  echo "  https://www.python.org/downloads/ から Python をインストールしてから、"
  echo "  このファイルをもう一度ダブルクリックしてください。"
  echo ""
  read -r -p "Enter キーで閉じます..."
  exit 1
fi

echo "Python: $(python3 --version)"
echo ""

if [ ! -d ".venv" ]; then
  echo "初回準備をしています（1〜3分ほどかかります）..."
  python3 -m venv .venv || { echo "【エラー】準備に失敗しました"; read -r -p "Enter キーで閉じます..."; exit 1; }
fi

echo "必要な部品を確認しています..."
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements.txt || {
  echo "【エラー】部品のインストールに失敗しました。上のメッセージを貼って報告してください。"
  read -r -p "Enter キーで閉じます..."
  exit 1
}

echo ""
echo "設定画面をブラウザに開きます。"
echo "（この黒い画面は開いたままにしておいてください。終わったら閉じて構いません）"
echo ""
PYTHONPATH=src ./.venv/bin/python -m short_trade setup

echo ""
read -r -p "Enter キーで閉じます..."
