@echo off
rem ユニバース拡張（1回だけ）。expand_windows.bat から呼ばれる。
cd /d "%~dp0\.."
chcp 65001 >nul
echo ===============================================
echo  short-trade ユニバース拡張（時点ユニバース 300銘柄）
echo ===============================================
call scripts\common.bat :ensure_python || goto :fail
call scripts\common.bat :self_update
call scripts\common.bat :ensure_venv || goto :fail
if not exist ".env" (
  echo.
  echo 【APIキーが未設定です】start_windows.bat を先に実行してください。
  goto :fail
)
set PYTHONPATH=src
echo.
echo 毎年の基準日ごとに「その時点の」プライム（旧・市場第一部）時価総額上位 300 銘柄を選び、和集合の日足を取得します（数分〜十数分）...
".venv\Scripts\python.exe" -m short_trade fetch --universe-pit --every 12 --market プライム --top 300 || goto :fail
echo.
echo 追加した銘柄の決算発表予定日を取得します...
".venv\Scripts\python.exe" -m short_trade fetch --earnings
echo.
echo 完了。以後は run_windows.bat を開けば 300 銘柄で比較が走ります（目安 10〜20 分）。
pause
exit /b 0
:fail
echo.
pause
exit /b 1
