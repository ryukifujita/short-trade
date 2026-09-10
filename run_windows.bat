@echo off
chcp 65001 >nul
rem short-trade 2回目以降（Windows 用）。決算日データが無ければ取得し、比較表を出します。
cd /d "%~dp0"
echo ===============================================
echo  short-trade 比較レポート
echo ===============================================
echo.
if not exist ".env" (
  echo 【先に start_windows.bat を実行してください】初回セットアップがまだです。
  pause & exit /b 1
)
set PYTHONPATH=src
if not exist "data\jquants\earnings.parquet" (
  echo 決算発表予定日を取得します（初回のみ。数分〜十数分かかります）...
  ".venv\Scripts\python.exe" -m short_trade fetch --earnings
  echo.
)
".venv\Scripts\python.exe" -m short_trade compare
echo.
echo ----- 戦略間の相関（Phase 2 の4本） -----
".venv\Scripts\python.exe" -m short_trade correlate
echo.
pause
