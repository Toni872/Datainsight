@echo off
echo ===================================================
echo  SOLUCIÓN DEFINITIVA PARA HTTPS EN WINDOWS
echo ===================================================
echo.

echo Este script utiliza una versión modificada del código que
echo resuelve los problemas de permisos en Windows para HTTPS
echo.

echo 1. Haciendo copia de seguridad del archivo app.ts original...
if not exist "src\app.ts.backup" (
    copy "src\app.ts" "src\app.ts.backup"
    echo Copia de seguridad guardada como src\app.ts.backup
) else (
    echo Ya existe una copia de seguridad, no se sobrescribirá
)

echo.
echo 2. Copiando la versión compatible con Windows...
copy "src\app-windows.ts" "src\app.ts" /Y
echo El archivo app.ts ha sido actualizado con una versión compatible con Windows

echo.
echo 3. Verificando certificados SSL...
if not exist ssl\fullchain.pem (
    echo Los certificados SSL no existen. Generando certificados autofirmados...
    call npm run generate-cert
) else (
    echo Certificados SSL encontrados correctamente.
)

echo.
echo 4. Iniciando el servidor...
echo El servidor estará disponible en: https://localhost:59999
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
echo Si deseas restaurar el archivo app.ts original, ejecuta:
echo copy "src\app.ts.backup" "src\app.ts" /Y
echo.
pause
