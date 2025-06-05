@echo off
echo -------------------------------------------
echo Iniciando DataInsight con HTTPS
echo -------------------------------------------

echo Sitio web disponible en:
echo HTTP: http://localhost:3000
echo HTTPS: https://localhost:3443
echo.

cd /d "%~dp0"
npm run dev
