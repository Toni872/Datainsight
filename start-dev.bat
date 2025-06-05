@echo off
echo Iniciando mi-proyecto en modo desarrollo
echo Puerto HTTP: 3000
echo Puerto HTTPS: 3443

cd /d "%~dp0"
npm run dev
