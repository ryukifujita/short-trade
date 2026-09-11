@echo off
chcp 65001 >nul
rem 初回セットアップ（Windows）。start_windows.bat から呼ばれる。
cd /d "%~dp0.."
echo ===============================================
echo  short-trade 初回セットアップ
echo ===============================================
call scripts\common.bat :ensure_python || exit /b 1
call scripts\common.bat :ensure_venv || exit /b 1
echo.
echo 設定画面をブラウザに開きます。（この黒い画面は開いたままにしてください）
echo.
set PYTHONPATH=src
".venv\Scripts\python.exe" -m short_trade setup
echo.
pause
