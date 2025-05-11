@echo off
echo ===================================================
echo       SOLUCIÓN PROBLEMAS HTTPS EN WINDOWS
echo ===================================================
echo.

echo Este script resuelve problemas comunes de acceso a HTTPS en desarrollo.
echo.

echo 1. Configurando Node.js para usar puertos sin privilegios
echo -----------------------------------------------------
echo.

echo Primero, eliminemos cualquier regla de firewall existente...
echo.
PowerShell -Command "Remove-NetFirewallRule -DisplayName \"Node.js HTTPS 59999\" -ErrorAction SilentlyContinue"

echo Ahora, creamos una nueva regla de firewall...
echo.
PowerShell -Command "New-NetFirewallRule -DisplayName \"Node.js HTTPS 59999\" -Direction Inbound -LocalPort 59999 -Protocol TCP -Action Allow | Out-Null"

echo 2. Asegurando que las variables de entorno estén correctamente definidas
echo -----------------------------------------------------
echo.

:: Modificar directamente el archivo .env
set ENV_FILE=%~dp0.env
if exist %ENV_FILE% (
    echo Actualizando variables de entorno...
    PowerShell -Command "(Get-Content %ENV_FILE%) | ForEach-Object { $_ -replace 'HTTPS_PORT=\d+', 'HTTPS_PORT=59999' } | Set-Content %ENV_FILE%"
    PowerShell -Command "(Get-Content %ENV_FILE%) | ForEach-Object { $_ -replace 'SSL_ENABLED=false', 'SSL_ENABLED=true' } | Set-Content %ENV_FILE%"
) else (
    echo No se encontró el archivo .env
)

echo 3. Iniciando el servidor
echo -----------------------------------------------------
echo.
echo El servidor se iniciará en: https://localhost:59999
echo NOTA: Es normal recibir una advertencia de seguridad en el navegador
echo       por el certificado autofirmado. Haz clic en "Continuar" o
echo       "Avanzado" para acceder al sitio.
echo.

:: Configurar temporalmente la variable de entorno SSL_ENABLED
set SSL_ENABLED=true

echo Iniciando el servidor HTTPS...
echo.
call npm run dev:ssl

echo.
echo Si sigues teniendo problemas, prueba a ejecutar este script como administrador.
echo.
pause
