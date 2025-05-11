@description('Nombre de la aplicación y los recursos asociados')
param applicationName string = 'datainsight'

@description('Ubicación para todos los recursos')
param location string = resourceGroup().location

@description('Tags para todos los recursos')
param resourceTags object = {
  Environment: 'Production'
  Project: 'DataInsight AI'
}

@description('SKU para App Service Plan')
param appServicePlanSku object = {
  name: 'P1v2'
  tier: 'PremiumV2'
  size: 'P1v2'
  family: 'Pv2'
  capacity: 1
}

@description('Nombre del entorno (usado para generar nombres únicos)')
param environmentName string

// Variables para nombres de recursos
var uniqueSuffix = uniqueString(resourceGroup().id)
var webAppName = '${applicationName}-web-${uniqueSuffix}'
var appServicePlanName = '${applicationName}-plan-${uniqueSuffix}'
var storageAccountName = replace('${applicationName}stor${uniqueSuffix}', '-', '')
var containerAppName = '${applicationName}-ml-${uniqueSuffix}'
var containerAppEnvName = '${applicationName}-env-${uniqueSuffix}'
var keyVaultName = '${applicationName}-kv-${uniqueSuffix}'
var logAnalyticsName = '${applicationName}-logs-${uniqueSuffix}'
var cosmosDbAccountName = '${applicationName}-cosmos-${uniqueSuffix}'

// Implementar el App Service para la aplicación web Node.js/TypeScript
module webApp 'app.bicep' = {
  name: 'webAppDeployment'
  params: {
    webAppName: webAppName
    appServicePlanName: appServicePlanName
    location: location
    appServicePlanSku: appServicePlanSku
    environmentName: environmentName
    resourceTags: resourceTags
    storageAccountName: storageAccountName
    keyVaultName: keyVaultName
  }
}

// Implementar recursos para Machine Learning
module mlResources 'ml.bicep' = {
  name: 'mlResourcesDeployment'
  params: {
    containerAppName: containerAppName
    containerAppEnvName: containerAppEnvName
    location: location
    environmentName: environmentName
    logAnalyticsName: logAnalyticsName
    resourceTags: resourceTags
    keyVaultName: keyVaultName
    storageAccountName: storageAccountName
  }
}

// Salidas para usar en scripts de despliegue
output webAppUrl string = webApp.outputs.webAppUrl
output webAppName string = webApp.outputs.webAppName
output storageAccountName string = storageAccountName
output containerAppName string = containerAppName
output keyVaultName string = keyVaultName