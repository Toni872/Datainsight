# Configuración del dominio datainsight.ai

Una vez que hayas adquirido el dominio datainsight.ai, sigue estos pasos para configurarlo con tu aplicación en Azure:

## 1. Configurar DNS en tu proveedor de dominio

Añade los siguientes registros DNS en tu proveedor de dominio (Namecheap, GoDaddy, etc.):

| Tipo    | Host     | Valor                                      | TTL      |
|---------|----------|-------------------------------------------|----------|
| A       | @        | [IP de tu Azure App Service]               | 600      |
| A       | www      | [IP de tu Azure App Service]               | 600      |
| CNAME   | api      | datainsight-webapp.azurewebsites.net       | 600      |
| CNAME   | ml       | datainsight-ml-service.azurecontainerapps.io | 600    |
| CNAME   | static   | datainsight-static.azureedge.net           | 600      |
| TXT     | @        | v=spf1 include:spf.protection.outlook.com -all | 600  |
| MX      | @        | 0 datainsight-ai.mail.protection.outlook.com | 600   |

Para obtener la IP de tu Azure App Service, ejecuta:
```powershell
az webapp show --resource-group datainsight-rg --name datainsight-webapp --query outboundIpAddresses --output tsv
```

## 2. Verificar la propiedad del dominio en Azure

### En Azure Portal:
1. Ve a tu App Service
2. Selecciona "Dominios personalizados"
3. Haz clic en "Agregar dominio personalizado"
4. Ingresa "www.datainsight.ai" y sigue las instrucciones para la verificación

### O usando Azure CLI:
```powershell
az webapp config hostname add --webapp-name datainsight-webapp --resource-group datainsight-rg --hostname "www.datainsight.ai"
```

## 3. Configurar SSL para tu dominio

### Opción 1: Certificado gestionado por Azure
```powershell
az webapp config ssl create --resource-group datainsight-rg --name datainsight-webapp --hostname "www.datainsight.ai"
```

### Opción 2: Certificado personalizado
Si ya tienes un certificado SSL:
```powershell
az webapp config ssl bind --certificate-thumbprint <CERT_THUMBPRINT> --ssl-type SNI --name datainsight-webapp --resource-group datainsight-rg
```

## 4. Configurar redirecciones de dominio

Las redirecciones ya están configuradas en tu archivo web.config para:
- Redirigir de datainsight.ai a www.datainsight.ai
- Redirigir de HTTP a HTTPS

## 5. Verificar la configuración

Una vez completados todos los pasos, verifica que tu sitio sea accesible a través de:
- https://www.datainsight.ai
- https://api.datainsight.ai
- https://ml.datainsight.ai
- https://static.datainsight.ai

## 6. Configurar email corporativo (opcional)

Si deseas configurar cuentas de correo con tu dominio (como info@datainsight.ai), puedes:
1. Usar Microsoft 365 Business Basic (aproximadamente 5€/usuario/mes)
2. Configurar Google Workspace (aproximadamente 6€/usuario/mes)
3. Usar servicios de email incluidos con tu proveedor de dominio
