@echo off
chcp 65001 >nul
rem 2回目以降（Windows）。run_windows.bat から呼ばれる。最新化 → 環境確認 → 決算日 → 比較表 → 相関
cd /d "%~dp0.."
echo ===============================================
echo  short-trade 比較レポート
echo ===============================================
call scripts\common.bat :self_update
call scripts\common.bat :ensure_python || exit /b 1
call scripts\common.bat :ensure_venv || exit /b 1
if not exist ".env" (
  echo.
  echo 【APIキーが未設定です】start_windows.bat を先に実行してください。
  pause & exit /b 1
)
set PYTHONPATH=src
if not exist "data\jquants\daily" (
  echo.
  echo 株価データがありません。まず取得します（30銘柄・数分）...
  ".venv\Scripts\python.exe" -m short_trade fetch --index topix
  ".venv\Scripts\python.exe" -m short_trade fetch --codes 7203,6758,9432,8058,4063,6098,6501,8035,9984,7974,6861,8306,4502,6902,7741,4568,6367,9433,8031,6594,4519,6954,7267,8766,6857,4661,9983,2914,8001,3382
)
if not exist "data\jquants\earnings.parquet" (
  echo.
  echo 決算発表予定日を取得します（初回のみ。数分〜十数分）...
  ".venv\Scripts\python.exe" -m short_trade fetch --earnings
)
echo.
".venv\Scripts\python.exe" -m short_trade compare
echo.
echo ----- 戦略間の相関（Phase 2 の4本） -----
".venv\Scripts\python.exe" -m short_trade correlate
echo.
pause
