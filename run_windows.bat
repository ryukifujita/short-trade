@echo off
rem 2回目以降（Windows）。中身は scripts\run.bat（起動時に自動で最新化）。このファイルは変わらない。
cd /d "%~dp0"
if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)
if exist "run_windows.new.bat" del /q "run_windows.new.bat"
call scripts\run.bat
