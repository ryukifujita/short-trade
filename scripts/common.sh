#!/bin/bash
# 起動ファイル共通処理（macOS / Linux）。scripts/ 配下は自己更新で置き換わる。
# 状態（.env / .venv / data/）には触らない。

BRANCH="claude/stock-trading-tool-requirements-d0irut"
ZIP_URL="https://github.com/ryukifujita/short-trade/archive/refs/heads/${BRANCH}.zip"
# 自己更新で置き換える対象（状態ファイルは含めない）
UPDATE_ITEMS="src catalog docs tests config scripts requirements.txt README.md start_mac.command run_mac.command start_windows.bat run_windows.bat .gitignore"

pause_exit() { echo ""; read -r -p "Enter キーで閉じます..."; exit "${1:-1}"; }

ensure_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "【エラー】Python が見つかりませんでした。"
    echo "  https://www.python.org/downloads/ からインストールして、もう一度開いてください。"
    pause_exit 1
  fi
  echo "Python: $(python3 --version)"
}

ensure_venv() {
  if [ ! -d ".venv" ]; then
    echo "初回準備をしています（1〜3分）..."
    python3 -m venv .venv || { echo "【エラー】準備に失敗しました"; pause_exit 1; }
  fi
  echo "必要な部品を確認しています..."
  ./.venv/bin/python -m pip install --quiet --upgrade pip >/dev/null 2>&1
  ./.venv/bin/python -m pip install --quiet -r requirements.txt || {
    echo "【エラー】部品のインストールに失敗しました。上のメッセージを貼って報告してください。"; pause_exit 1; }
}

self_update() {
  # GitHub から最新のコードを取り、状態ファイル以外を入れ替える。失敗しても続行する。
  command -v curl >/dev/null 2>&1 || { echo "（curl が無いため更新をスキップ）"; return 0; }
  local tmp; tmp="$(mktemp -d)" || return 0
  echo "コードを最新にしています..."
  if curl -fsSL --max-time 90 "$ZIP_URL" -o "$tmp/code.zip" && unzip -q "$tmp/code.zip" -d "$tmp"; then
    local src; src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'short-trade-*' | head -1)"
    if [ -n "$src" ]; then
      for item in $UPDATE_ITEMS; do
        if [ -e "$src/$item" ]; then
          rm -rf "./$item.new"
          cp -R "$src/$item" "./$item.new"
          rm -rf "./$item"
          mv "./$item.new" "./$item"      # rename は実行中のスクリプトにも安全
        fi
      done
      chmod +x ./*.command ./scripts/*.sh 2>/dev/null
      echo "コードを最新にしました（APIキー・データ・環境はそのまま）"
    fi
  else
    echo "（更新をスキップ: ネットワークに繋がらないか、ダウンロードに失敗。手元のコードで続けます）"
  fi
  rm -rf "$tmp"
}
