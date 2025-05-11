@echo off
echo -------------------------------------------
echo Configurando y ejecutando servidor con HTTPS
echo -------------------------------------------

echo 1. Creando reserva de puerto para NodeJS
:: Este comando requiere privilegios de administrador
echo Habilitar NodeJS para usar el puerto 45678 sin privilegios de administrador
netsh http add urlacl url=https://+:45678/ user=Everyone

echo.
echo 2. Verificando certificados SSL
if not exist ssl\fullchain.pem (
    echo Los certificados SSL no existen. Generando certificados autofirmados...
    call npm run generate-cert
) else (
    echo Los certificados SSL ya existen.
)

echo.
echo 3. Iniciando servidor con HTTPS habilitado
echo Servidor iniciándose en https://localhost:45678
call npm run dev:ssl

echo.
echo Si el servidor no inicia correctamente, prueba ejecutando este script como administrador.
pause
