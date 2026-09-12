@echo off
rem パラメータ探索（学習／検証の分割つき）。tune_windows.bat から呼ばれる。時間がかかる（30〜60分）。
cd /d "%~dp0\.."
chcp 65001 >nul
echo ===============================================
echo  short-trade パラメータ探索（学習 〜2021-12-31 / 検証 2022 以降）
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
echo 宣言済みのパラメータ（各 3 水準）を学習期間で探索し、検証期間で既定値と比べます（30〜60 分）...
".venv\Scripts\python.exe" -m short_trade optimize --train-end 2021-12-31
pause
exit /b 0
:fail
echo.
pause
exit /b 1
