function Install-GnuTLS {
    param(
        [string]$SourceZipFile = "gnutls.v2.2.9.1701.x64.zip",
        [string]$DestinationPath = "C:\Program Files\GnuTLS"
    )

    Write-Host "Instalando GnuTLS en Windows..." -ForegroundColor Green

    # Verificar que el archivo existe
    if (-not (Test-Path $SourceZipFile)) {
        Write-Host "ERROR: El archivo $SourceZipFile no existe en la ubicación actual." -ForegroundColor Red
        Write-Host "Por favor, coloca el archivo zip descargado en la misma carpeta que este script." -ForegroundColor Yellow
        return
    }

    # Crear directorio de destino si no existe
    if (-not (Test-Path $DestinationPath)) {
        Write-Host "Creando directorio $DestinationPath..." -ForegroundColor Yellow
        New-Item -Path $DestinationPath -ItemType Directory -Force | Out-Null
    }

    # Extraer archivo ZIP
    Write-Host "Extrayendo archivos de $SourceZipFile a $DestinationPath..." -ForegroundColor Yellow
    Expand-Archive -Path $SourceZipFile -DestinationPath $DestinationPath -Force

    # Añadir al PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Machine)
    $binPath = Join-Path $DestinationPath "bin"
    
    if (-not $currentPath.Contains($binPath)) {
        Write-Host "Añadiendo $binPath al PATH del sistema..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$binPath", [EnvironmentVariableTarget]::Machine)
        $env:PATH = "$env:PATH;$binPath"
        Write-Host "PATH actualizado. Es posible que necesites reiniciar tu terminal." -ForegroundColor Green
    } else {
        Write-Host "GnuTLS ya está en el PATH del sistema." -ForegroundColor Green
    }

    Write-Host "Instalación completa. Probando gnutls-cli..." -ForegroundColor Green
    try {
        $gnutlsVersion = & "$binPath\gnutls-cli.exe" --version 2>&1
        Write-Host $gnutlsVersion -ForegroundColor Cyan
        Write-Host "GnuTLS instalado correctamente." -ForegroundColor Green
    } catch {
        Write-Host "No se pudo ejecutar gnutls-cli. Es posible que necesites reiniciar tu terminal." -ForegroundColor Yellow
    }
}

# Llamar a la función con la ubicación del archivo ZIP
$zipPath = Read-Host "Introduce la ruta completa al archivo gnutls.v2.2.9.1701.x64.zip (o pulsa Enter para usar la ubicación actual)"
if ([string]::IsNullOrEmpty($zipPath)) {
    $zipPath = "gnutls.v2.2.9.1701.x64.zip"
}

Install-GnuTLS -SourceZipFile $zipPath

