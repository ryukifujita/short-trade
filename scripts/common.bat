@echo off
rem 共通処理（Windows）。call scripts\common.bat :ラベル で呼ぶ。
set "BRANCH=claude/stock-trading-tool-requirements-d0irut"
set "ZIP_URL=https://github.com/ryukifujita/short-trade/archive/refs/heads/%BRANCH%.zip"
if "%~1"=="" exit /b 0
call %~1
exit /b %errorlevel%

:ensure_python
where python >nul 2>&1
if errorlevel 1 (
  echo 【エラー】Python が見つかりません。https://www.python.org/downloads/ からインストールし、
  echo 「Add python.exe to PATH」にチェックを入れてください。
  pause & exit /b 1
)
for /f "delims=" %%v in ('python --version') do echo Python: %%v
exit /b 0

:ensure_venv
if not exist ".venv" (
  echo 初回準備をしています（1〜3分）...
  python -m venv .venv || (echo 【エラー】準備に失敗しました & pause & exit /b 1)
)
echo 必要な部品を確認しています...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || (echo 【エラー】部品のインストールに失敗しました & pause & exit /b 1)
exit /b 0

:self_update
rem GitHub から最新コードを取り、状態ファイル（.env / .venv / data）以外を入れ替える。失敗しても続行。
set "TMPD=%TEMP%\short-trade-update-%RANDOM%"
mkdir "%TMPD%" >nul 2>&1
echo コードを最新にしています...
curl -fsSL --max-time 90 "%ZIP_URL%" -o "%TMPD%\code.zip" >nul 2>&1
if errorlevel 1 (echo （更新をスキップ: ダウンロードに失敗。手元のコードで続けます） & rmdir /s /q "%TMPD%" & exit /b 0)
tar -xf "%TMPD%\code.zip" -C "%TMPD%" >nul 2>&1
if errorlevel 1 (echo （更新をスキップ: 展開に失敗） & rmdir /s /q "%TMPD%" & exit /b 0)
for /d %%d in ("%TMPD%\short-trade-*") do set "SRC=%%d"
if not defined SRC (rmdir /s /q "%TMPD%" & exit /b 0)
for %%i in (src catalog docs tests config requirements.txt README.md start_mac.command run_mac.command expand_mac.command tune_mac.command start_windows.bat expand_windows.bat tune_windows.bat .gitignore) do (
  if exist "%SRC%\%%i\" (rmdir /s /q "%%i" >nul 2>&1 & xcopy /E /I /Q /Y "%SRC%\%%i" "%%i" >nul)
  if exist "%SRC%\%%i" if not exist "%SRC%\%%i\" copy /Y "%SRC%\%%i" "%%i" >nul
)
rem scripts\ は実行中のため、次回起動時に入れ替わるよう .new に置く
if exist "%SRC%\scripts" (rmdir /s /q "scripts.new" >nul 2>&1 & xcopy /E /I /Q /Y "%SRC%\scripts" "scripts.new" >nul)
if exist "%SRC%\run_windows.bat" copy /Y "%SRC%\run_windows.bat" "run_windows.new.bat" >nul
rmdir /s /q "%TMPD%"
echo コードを最新にしました（APIキー・データ・環境はそのまま。scripts は次回起動時に反映）
exit /b 0
