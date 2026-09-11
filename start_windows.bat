@echo off
rem 初回セットアップ（Windows）。中身は scripts\setup.bat。このファイルは変わらない。
cd /d "%~dp0"
if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)
call scripts\setup.bat
