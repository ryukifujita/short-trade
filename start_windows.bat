@echo off
chcp 65001 >nul
rem short-trade セットアップ（Windows 用）
rem このファイルをダブルクリックすると、必要な準備をして設定画面をブラウザに開きます。

cd /d "%~dp0"
echo ===============================================
echo  short-trade セットアップ
echo ===============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo 【エラー】Python が見つかりませんでした。
  echo.
  echo   https://www.python.org/downloads/ から Python をインストールしてください。
  echo   インストール画面の一番下にある「Add python.exe to PATH」に
  echo   必ずチェックを入れてください。
  echo.
  echo   インストール後、このファイルをもう一度ダブルクリックしてください。
  echo.
  pause
  exit /b 1
)

for /f "delims=" %%v in ('python --version') do echo Python: %%v
echo.

if not exist ".venv" (
  echo 初回準備をしています（1〜3分ほどかかります）...
  python -m venv .venv
  if errorlevel 1 (
    echo 【エラー】準備に失敗しました
    pause
    exit /b 1
  )
)

echo 必要な部品を確認しています...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo 【エラー】部品のインストールに失敗しました。上のメッセージを貼って報告してください。
  pause
  exit /b 1
)

echo.
echo 設定画面をブラウザに開きます。
echo （この黒い画面は開いたままにしておいてください。終わったら閉じて構いません）
echo.
set PYTHONPATH=src
".venv\Scripts\python.exe" -m short_trade setup

echo.
pause
