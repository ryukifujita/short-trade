@echo off
rem パラメータ探索（Windows）。中身は scripts\tune.bat。このファイルは変わらない。
cd /d "%~dp0"
if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)
call scripts\tune.bat
