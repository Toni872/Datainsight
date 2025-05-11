@description('Nombre de la aplicación web')
param webAppName string

@description('Nombre del plan de App Service')
param appServicePlanName string

@description('Ubicación para todos los recursos')
param location string

@description('SKU para App Service Plan')
param appServicePlanSku object

@description('Nombre del entorno (usado para generar nombres únicos)')
param environmentName string

@description('Tags para todos los recursos')
param resourceTags object

@description('Nombre de la cuenta de almacenamiento')
param storageAccountName string

@description('Nombre del Key Vault')
param keyVaultName string

// Definición de la identidad administrada para el App Service
var managedIdentityName = '${webAppName}-id'

// Crear la identidad administrada para el App Service
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: managedIdentityName
  location: location
  tags: resourceTags
}

// Crear el plan de App Service
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: appServicePlanName
  location: location
  tags: resourceTags
  sku: {
    name: appServicePlanSku.name
    tier: appServicePlanSku.tier
    size: appServicePlanSku.size
    family: appServicePlanSku.family
    capacity: appServicePlanSku.capacity
  }
  properties: {
    reserved: false // true para Linux
  }
}

// Crear la cuenta de almacenamiento para archivos estáticos y datos
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: resourceTags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    accessTier: 'Hot'
  }
}

// Crear el contenedor de blobs para datasets
resource datasetsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/datasets'
  properties: {
    publicAccess: 'None'
  }
}

// Crear el contenedor de blobs para uploads temporales
resource uploadsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/uploads'
  properties: {
    publicAccess: 'None'
  }
}

// Crear Key Vault para secretos
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: resourceTags
  properties: {
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    sku: {
      name: 'standard'
      family: 'A'
    }
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Crear la aplicación web
resource webApp 'Microsoft.Web/sites@2022-09-01' = {
  name: webAppName
  location: location
  tags: union(resourceTags, {
    'azd-env-name': environmentName
    'azd-service-name': 'web'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      appSettings: [
        {
          name: 'NODE_ENV'
          value: 'production'
        }
        {
          name: 'DOMAIN'
          value: 'www.contactodatainsight.com'
        }
        {
          name: 'STORAGE_CONNECTION_STRING'
          value: '@Microsoft.KeyVault(SecretUri=${keyVault.properties.vaultUri}secrets/StorageConnectionString)'
        }
        {
          name: 'MANAGED_IDENTITY_CLIENT_ID'
          value: managedIdentity.properties.clientId
        }
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: 'true'
        }
      ]
      cors: {
        allowedOrigins: [
          'https://www.contactodatainsight.com'
        ]
      }
      nodeVersion: '~20'
    }
  }
}

// Crear configuración de diagnóstico para la aplicación web
resource webAppDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${webAppName}-diagnostics'
  scope: webApp
  properties: {
    logs: [
      {
        category: 'AppServiceHTTPLogs'
        enabled: true
        retentionPolicy: {
          days: 30
          enabled: true
        }
      }
      {
        category: 'AppServiceConsoleLogs'
        enabled: true
        retentionPolicy: {
          days: 30
          enabled: true
        }
      }
      {
        category: 'AppServiceAppLogs'
        enabled: true
        retentionPolicy: {
          days: 30
          enabled: true
        }
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
        retentionPolicy: {
          days: 30
          enabled: true
        }
      }
    ]
  }
}

// Crear configuración de dominio personalizado
resource customDomain 'Microsoft.Web/sites/hostNameBindings@2022-09-01' = {
  name: '${webApp.name}/www.contactodatainsight.com'
  properties: {
    hostNameType: 'Verified'
    sslState: 'Disabled'
    customHostNameDnsRecordType: 'CName'
  }
}

// Salidas
output webAppUrl string = 'https://${webApp.properties.defaultHostName}'
output webAppName string = webApp.name
