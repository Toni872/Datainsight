# Script para desplegar DataInsight AI en Azure
# Asegúrate de haber iniciado sesión con 'az login' antes de ejecutar este script

# Variables de configuración
$resourceGroup = "datainsight-rg"
$location = "westeurope"
$webAppName = "datainsight-webapp"
$mlServiceName = "datainsight-ml-service"
$cosmosDBName = "datainsight-cosmos"
$storageName = "datainsightstorage"
$cdnProfileName = "datainsight-cdn"
$cdnEndpointName = "datainsight-static"

Write-Host "=== Iniciando despliegue de DataInsight AI en Azure ===" -ForegroundColor Green

# 1. Crear grupo de recursos
Write-Host "Creando grupo de recursos..." -ForegroundColor Cyan
az group create --name $resourceGroup --location $location

# 2. Crear plan de App Service
Write-Host "Creando plan de App Service..." -ForegroundColor Cyan
az appservice plan create --name "datainsight-plan" --resource-group $resourceGroup --sku B1 --is-linux

# 3. Crear Web App para la aplicación Node.js
Write-Host "Creando Web App para Node.js..." -ForegroundColor Cyan
az webapp create --resource-group $resourceGroup --plan "datainsight-plan" --name $webAppName --runtime "NODE:18-lts"

# 4. Configurar variables de entorno para la Web App
Write-Host "Configurando variables de entorno para la Web App..." -ForegroundColor Cyan
az webapp config appsettings set --resource-group $resourceGroup --name $webAppName --settings @.env.production

# 5. Crear Azure Container Registry para el servicio ML
Write-Host "Creando Azure Container Registry..." -ForegroundColor Cyan
az acr create --resource-group $resourceGroup --name "datainsightacr" --sku Basic --admin-enabled true

# 6. Construir y subir imagen Docker para el servicio ML
Write-Host "Construyendo y subiendo imagen Docker para el servicio ML..." -ForegroundColor Cyan
az acr build --registry "datainsightacr" --image "ml-service:latest" ./data_science_ml_learning

# 7. Crear Container App para el servicio ML
Write-Host "Creando Container App para el servicio ML..." -ForegroundColor Cyan
az containerapp create --resource-group $resourceGroup --name $mlServiceName --image "datainsightacr.azurecr.io/ml-service:latest" --target-port 8000 --ingress external

# 8. Crear base de datos Cosmos DB con API para MongoDB
Write-Host "Creando Cosmos DB con API para MongoDB..." -ForegroundColor Cyan
az cosmosdb create --name $cosmosDBName --resource-group $resourceGroup --kind MongoDB

# 9. Crear cuenta de almacenamiento para archivos estáticos
Write-Host "Creando cuenta de almacenamiento..." -ForegroundColor Cyan
az storage account create --name $storageName --resource-group $resourceGroup --location $location --sku Standard_LRS --kind StorageV2

# 10. Crear perfil CDN y endpoint para archivos estáticos
Write-Host "Creando perfil CDN y endpoint..." -ForegroundColor Cyan
az cdn profile create --name $cdnProfileName --resource-group $resourceGroup --sku Standard_Microsoft
az cdn endpoint create --name $cdnEndpointName --profile-name $cdnProfileName --resource-group $resourceGroup --origin $storageName.blob.core.windows.net

# 11. Configurar dominio personalizado
Write-Host "NOTA: Para configurar el dominio personalizado (datainsight.ai), sigue estos pasos:" -ForegroundColor Yellow
Write-Host "1. Añade un registro A en tu proveedor de dominio apuntando a la IP de tu Web App" -ForegroundColor Yellow
Write-Host "2. Ejecuta: az webapp config hostname add --webapp-name $webAppName --resource-group $resourceGroup --hostname 'www.datainsight.ai'" -ForegroundColor Yellow
Write-Host "3. Configura SSL con: az webapp config ssl bind --certificate-thumbprint \$CERT_THUMBPRINT --ssl-type SNI --name $webAppName --resource-group $resourceGroup" -ForegroundColor Yellow

Write-Host "=== Despliegue completado! ===" -ForegroundColor Green
Write-Host "URL de la aplicación web: https://$webAppName.azurewebsites.net" -ForegroundColor Cyan
Write-Host "URL del servicio ML: https://$mlServiceName.azurecontainerapps.io" -ForegroundColor Cyan
