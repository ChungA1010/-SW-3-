@echo off
setlocal
cd /d "%~dp0frontend"
call "C:\Program Files\nodejs\npm.cmd" run dev
