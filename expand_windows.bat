@echo off
rem ユニバース拡張（Windows、1回だけ）。中身は scripts\expand.bat。このファイルは変わらない。
cd /d "%~dp0"
if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)
call scripts\expand.bat
