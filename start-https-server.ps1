# Script para iniciar un servidor HTTPS en desarrollo con Node.js
# Resuelve problemas comunes de permisos de puerto y certificados

# Colores para mensajes
$infoColor = "Cyan"
$successColor = "Green"
$warningColor = "Yellow"
$errorColor = "Red"
$highlightColor = "Magenta"

function Test-IsAdmin {
    return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-PortAvailable {
    param ([int]$Port)
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

# Mensaje de bienvenida
Clear-Host
Write-Host "====================================================" -ForegroundColor $highlightColor
Write-Host "        CONFIGURACIÓN DE SERVIDOR HTTPS LOCAL        " -ForegroundColor $highlightColor
Write-Host "====================================================" -ForegroundColor $highlightColor
Write-Host ""

# Comprobar si se está ejecutando como administrador
$isAdmin = Test-IsAdmin
Write-Host "Ejecutando como administrador: " -NoNewline
if ($isAdmin) {
    Write-Host "SÍ" -ForegroundColor $successColor
} else {
    Write-Host "NO" -ForegroundColor $warningColor
    Write-Host "NOTA: Algunas operaciones pueden requerir privilegios de administrador." -ForegroundColor $warningColor
}
Write-Host ""

# Verificar puerto HTTPS (12443)
Write-Host "Verificando disponibilidad del puerto 12443..." -ForegroundColor $infoColor
$portAvailable = Test-PortAvailable -Port 12443
if ($portAvailable) {
    Write-Host "El puerto 12443 está disponible." -ForegroundColor $successColor
} else {
    Write-Host "ADVERTENCIA: El puerto 12443 está en uso." -ForegroundColor $warningColor
    Write-Host "Puede que necesites terminar el proceso que lo está usando o cambiar el puerto en el archivo .env" -ForegroundColor $warningColor
    
    $continue = Read-Host "¿Desea intentar continuar de todos modos? (S/N)"
    if ($continue -ne "S" -and $continue -ne "s") {
        Write-Host "Operación cancelada por el usuario." -ForegroundColor $errorColor
        exit
    }
}
Write-Host ""

# Verificar certificados SSL
Write-Host "Verificando certificados SSL..." -ForegroundColor $infoColor
$sslDir = Join-Path $PSScriptRoot "ssl"
$privateKeyPath = Join-Path $sslDir "privkey.pem"
$certPath = Join-Path $sslDir "fullchain.pem"

if (-not (Test-Path $sslDir)) {
    Write-Host "Creando directorio SSL..." -ForegroundColor $warningColor
    New-Item -ItemType Directory -Path $sslDir -Force | Out-Null
}

$genCerts = $false
if (-not (Test-Path $privateKeyPath) -or -not (Test-Path $certPath)) {
    Write-Host "Certificados SSL no encontrados." -ForegroundColor $warningColor
    $genCerts = $true
} else {
    Write-Host "Certificados SSL encontrados." -ForegroundColor $successColor
    
    # Verificar fechas de los certificados
    try {
        $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 $certPath
        $expirationDate = $cert.NotAfter
        $daysRemaining = ($expirationDate - (Get-Date)).Days
        
        Write-Host "  - Válido hasta: $expirationDate" -ForegroundColor $infoColor
        if ($daysRemaining -lt 30) {
            Write-Host "  - ADVERTENCIA: El certificado expirará en $daysRemaining días." -ForegroundColor $warningColor
            
            $regenerate = Read-Host "¿Desea regenerar los certificados? (S/N)"
            if ($regenerate -eq "S" -or $regenerate -eq "s") {
                $genCerts = $true
            }
        } else {
            Write-Host "  - Días restantes: $daysRemaining" -ForegroundColor $successColor
        }
    } catch {
        Write-Host "Error al verificar el certificado existente. Puede estar corrupto." -ForegroundColor $warningColor
        $genCerts = $true
    }
}

# Generar certificados si es necesario
if ($genCerts) {
    Write-Host "Generando nuevos certificados SSL..." -ForegroundColor $highlightColor
    npm run generate-cert
    
    if (-not (Test-Path $privateKeyPath) -or -not (Test-Path $certPath)) {
        Write-Host "ERROR: No se pudieron generar los certificados SSL." -ForegroundColor $errorColor
        exit 1
    } else {
        Write-Host "Certificados SSL generados correctamente." -ForegroundColor $successColor
    }
}
Write-Host ""

# Reservar puerto para uso sin privilegios de administrador
if ($isAdmin) {
    Write-Host "Configurando reserva de puerto para NodeJS..." -ForegroundColor $infoColor
    try {
        $output = netsh http add urlacl url=https://+:12443/ user=Everyone
        Write-Host "Reserva de puerto configurada correctamente." -ForegroundColor $successColor
    } catch {
        Write-Host "Error al configurar la reserva de puerto: $_" -ForegroundColor $errorColor
    }
} else {
    Write-Host "ADVERTENCIA: Se necesitan privilegios de administrador para configurar la reserva de puerto." -ForegroundColor $warningColor
    Write-Host "Puede que el servidor no inicie correctamente sin esta configuración." -ForegroundColor $warningColor
}
Write-Host ""

# Iniciar servidor
Write-Host "Iniciando servidor con HTTPS habilitado..." -ForegroundColor $highlightColor
Write-Host "Servidor disponible en:" -ForegroundColor $infoColor
Write-Host "  - HTTP:  http://localhost:3000" -ForegroundColor $infoColor
Write-Host "  - HTTPS: https://localhost:12443" -ForegroundColor $highlightColor
Write-Host ""
Write-Host "NOTA: Al acceder por HTTPS, el navegador mostrará una advertencia de seguridad" -ForegroundColor $warningColor
Write-Host "      porque el certificado es autofirmado. Deberás aceptar manualmente." -ForegroundColor $warningColor
Write-Host ""

# Establecer la variable de entorno SSL_ENABLED a true para esta sesión
$env:SSL_ENABLED = "true"

# Ejecutar el servidor
npm run dev:ssl
