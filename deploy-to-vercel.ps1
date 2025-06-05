# Script de despliegue para Vercel
# Ejecutar desde PowerShell con permisos de administrador

# Colores para los mensajes
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow
$RED = [System.ConsoleColor]::Red
$CYAN = [System.ConsoleColor]::Cyan

# Función para mostrar mensajes con colores
function Write-ColorOutput($color, $message) {
    Write-Host $message -ForegroundColor $color
}

# Verificar si Vercel CLI está instalado
Write-ColorOutput $CYAN "Verificando si Vercel CLI está instalado..."
$vercelInstalled = $null
try {
    $vercelInstalled = Get-Command vercel -ErrorAction SilentlyContinue
} catch {
    $vercelInstalled = $null
}

if ($null -eq $vercelInstalled) {
    Write-ColorOutput $YELLOW "Vercel CLI no está instalado. Instalando..."
    npm install -g vercel
} else {
    Write-ColorOutput $GREEN "Vercel CLI ya está instalado."
}

# Construir el proyecto
Write-ColorOutput $CYAN "Construyendo el proyecto..."
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput $RED "Error durante la construcción. Abortando."
    exit 1
}

# Iniciar sesión en Vercel (si es necesario)
Write-ColorOutput $CYAN "Verificando la sesión de Vercel..."
vercel whoami

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput $YELLOW "No has iniciado sesión en Vercel. Iniciando sesión..."
    vercel login
}

# Configurar variables de entorno
Write-ColorOutput $CYAN "Configurando variables de entorno..."
Write-ColorOutput $YELLOW "¿Deseas configurar las variables de entorno ahora? (S/N)"
$configEnv = Read-Host

if ($configEnv -eq "S" -or $configEnv -eq "s") {
    # Leer variables del archivo .env.example
    if (Test-Path ".env.example") {
        Write-ColorOutput $GREEN "Usando .env.example como referencia para las variables..."
        $envVars = Get-Content ".env.example" | Where-Object { $_ -match '^[A-Za-z0-9_]+=.+' }
        
        foreach ($line in $envVars) {
            $varName = $line.Split('=')[0]
            
            # Ignorar comentarios y variables vacías
            if ($varName -match '^#' -or $varName -eq '') {
                continue
            }
            
            Write-ColorOutput $YELLOW "Configurar $varName (deja vacío para omitir):"
            $varValue = Read-Host
            
            if ($varValue -ne '') {
                vercel env add $varName
            }
        }
    } else {
        Write-ColorOutput $YELLOW "No se encontró el archivo .env.example. Configurando variables esenciales."
        
        Write-ColorOutput $YELLOW "MONGODB_URI:"
        vercel env add MONGODB_URI
        
        Write-ColorOutput $YELLOW "ML_SERVICE_URL:"
        vercel env add ML_SERVICE_URL
        
        Write-ColorOutput $YELLOW "JWT_SECRET:"
        vercel env add JWT_SECRET
        
        Write-ColorOutput $YELLOW "STRIPE_SECRET_KEY (opcional):"
        $stripeKey = Read-Host
        if ($stripeKey -ne '') {
            vercel env add STRIPE_SECRET_KEY
        }
    }
}

# Desplegar a Vercel
Write-ColorOutput $CYAN "¿Deseas desplegar ahora a Vercel? (S/N)"
$deploy = Read-Host

if ($deploy -eq "S" -or $deploy -eq "s") {
    Write-ColorOutput $CYAN "Desplegando a Vercel..."
    vercel --prod
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput $GREEN "¡Despliegue exitoso!"
        
        # Configurar dominio personalizado
        Write-ColorOutput $CYAN "¿Quieres configurar un dominio personalizado? (S/N)"
        $configDomain = Read-Host
        
        if ($configDomain -eq "S" -or $configDomain -eq "s") {
            Write-ColorOutput $YELLOW "Ingresa el dominio (ej. datainsight.es):"
            $domain = Read-Host
            
            if ($domain -ne '') {
                vercel domains add $domain
            }
        }
    } else {
        Write-ColorOutput $RED "Error durante el despliegue."
    }
} else {
    Write-ColorOutput $YELLOW "Puedes desplegar más tarde usando 'vercel --prod'"
}

Write-ColorOutput $GREEN "¡Proceso de preparación completado!"
Write-ColorOutput $CYAN "Recuerda configurar tu dominio y DNS según la documentación en docs/vercel-deployment.md"
