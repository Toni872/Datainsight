# Ejecutar con privilegios de administrador para configurar correctamente el servidor HTTPS

# Chequeamos si estamos ejecutando como administrador
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Este script requiere privilegios de administrador para configurar correctamente el puerto HTTPS." -ForegroundColor Yellow
    Write-Host "Por favor, ejecuta PowerShell como administrador y vuelve a ejecutar este script." -ForegroundColor Yellow
    Write-Host "Presiona cualquier tecla para salir..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Definir el puerto HTTPS a utilizar (usando un puerto muy alto para evitar problemas de permisos)
$httpsPort = 59999

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "      Configuración de Servidor HTTPS Local" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Paso 1: Verificar certificados SSL
Write-Host "1. Verificando certificados SSL..." -ForegroundColor Green

$sslDir = Join-Path $PSScriptRoot "ssl"
$privKeyPath = Join-Path $sslDir "privkey.pem"
$certPath = Join-Path $sslDir "fullchain.pem"

if (-not (Test-Path $privKeyPath) -or -not (Test-Path $certPath)) {
    Write-Host "   No se encontraron certificados SSL. Generando certificados autofirmados..." -ForegroundColor Yellow
    npm run generate-cert
} else {
    Write-Host "   Certificados SSL encontrados correctamente." -ForegroundColor Green
}

# Paso 2: Configurar reserva de URL para el puerto HTTPS
Write-Host ""
Write-Host "2. Configurando permisos para el puerto HTTPS ($httpsPort)..." -ForegroundColor Green

# Abrir el puerto en el firewall (enfoque más confiable en Windows)
try {
    # Verificar si ya existe una regla y eliminarla
    $existingRule = Get-NetFirewallRule -DisplayName "Node.js HTTPS $httpsPort" -ErrorAction SilentlyContinue
    if ($existingRule) {
        Write-Host "   Eliminando regla de firewall existente..." -ForegroundColor Cyan
        Remove-NetFirewallRule -DisplayName "Node.js HTTPS $httpsPort" -ErrorAction SilentlyContinue
    }
    
    # Crear nueva regla
    Write-Host "   Creando regla de firewall para el puerto $httpsPort..." -ForegroundColor Cyan
    New-NetFirewallRule -DisplayName "Node.js HTTPS $httpsPort" -Direction Inbound -LocalPort $httpsPort -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null
    Write-Host "   Puerto $httpsPort abierto en el firewall." -ForegroundColor Green
    
    # Intentar también el método de netsh como respaldo
    try {
        $urlReservation = "https://+:$httpsPort/"
        netsh http delete urlacl url=$urlReservation 2>$null
        netsh http add urlacl url=$urlReservation user=Everyone 2>$null
    } catch {
        # Silenciar errores del método netsh ya que tenemos el firewall configurado
    }
    
    Write-Host "   Permisos de puerto configurados correctamente." -ForegroundColor Green
} catch {
    Write-Host "   Advertencia al configurar el firewall: $_" -ForegroundColor Yellow
    Write-Host "   Continuando de todos modos, puede que funcione sin esta configuración..." -ForegroundColor Yellow
}

# Paso 3: Verificar las variables de entorno
Write-Host ""
Write-Host "3. Configurando variables de entorno..." -ForegroundColor Green

$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    $envContent = Get-Content $envFile -Raw
    
    # Verificar si el puerto HTTPS es correcto
    if ($envContent -notmatch "HTTPS_PORT=$httpsPort") {
        Write-Host "   Actualizando el puerto HTTPS en .env..." -ForegroundColor Yellow
        $envContent = $envContent -replace "HTTPS_PORT=\d+", "HTTPS_PORT=$httpsPort"
        $envContent | Set-Content $envFile -NoNewline
    }
    
    # Asegurarse de que SSL_ENABLED está en false por defecto (se activará mediante variables de entorno al ejecutar)
    if ($envContent -match "SSL_ENABLED=true") {
        Write-Host "   Configurando SSL_ENABLED como false por defecto en .env..." -ForegroundColor Yellow
        $envContent = $envContent -replace "SSL_ENABLED=true", "SSL_ENABLED=false"
        $envContent | Set-Content $envFile -NoNewline
    }
    
    Write-Host "   Variables de entorno configuradas correctamente." -ForegroundColor Green
} else {
    Write-Host "   No se encontró el archivo .env" -ForegroundColor Red
}

# Paso 4: Iniciar el servidor
Write-Host ""
Write-Host "4. Iniciando servidor con HTTPS habilitado..." -ForegroundColor Green
Write-Host "   El servidor estará disponible en: https://localhost:$httpsPort" -ForegroundColor Cyan
Write-Host "   Recuerda que necesitarás aceptar el certificado autofirmado en tu navegador." -ForegroundColor Yellow
Write-Host ""
Write-Host "   Presiona Ctrl+C para detener el servidor cuando hayas terminado." -ForegroundColor Magenta
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Establecer la variable de entorno SSL_ENABLED para esta sesión
$env:SSL_ENABLED = "true"

# Ejecutar el servidor
npm run dev:ssl
