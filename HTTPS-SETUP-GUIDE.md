# Guía de Configuración SSL/HTTPS para DataInsight

## Índice

1. [Instalación de GnuTLS](#instalación-de-gnutls)
2. [Configuración para Desarrollo Local](#configuración-para-desarrollo-local)
3. [Configuración para Producción](#configuración-para-producción)
4. [Solución de problemas comunes](#solución-de-problemas-comunes)

## Instalación de GnuTLS

El archivo que has descargado (`gnutls.v2.2.9.1701.x64.zip`) contiene las bibliotecas de GnuTLS para Windows, que son útiles para generar y manejar certificados SSL.

### Pasos para instalar GnuTLS

1. **Ejecuta el script de instalación con PowerShell como administrador**:

   ```powershell
   # Abre PowerShell como administrador y ejecuta:
   Set-ExecutionPolicy Bypass -Scope Process -Force
   ./InstallGnuTLS.ps1
   ```

2. **Indica la ruta al archivo zip de GnuTLS** cuando el script lo solicite.

3. **Reinicia tu terminal** para que los cambios en el PATH surtan efecto.

4. **Verifica la instalación**:
   ```powershell
   gnutls-cli --version
   ```

## Configuración para Desarrollo Local

Para ejecutar la aplicación localmente con HTTPS:

### Método 1: Usando el script automatizado

1. **Ejecuta el script batch como administrador**:
   ```
   # Clic derecho en start-https-server.bat -> Ejecutar como administrador
   ```

### Método 2: Manual

1. **Genera certificados SSL** (si no existen):
   ```bash
   npm run generate-cert
   ```

2. **Reserva el puerto para uso sin administrador** (requiere permisos de administrador una vez):
   ```powershell
   netsh http add urlacl url=https://+:12443/ user=Everyone
   ```

3. **Inicia el servidor con SSL habilitado**:
   ```bash
   npm run dev:ssl
   ```

4. **Accede a la aplicación** en tu navegador:
   ```
   https://localhost:12443
   ```

5. **Acepta el certificado autofirmado** en tu navegador (aparecerá una advertencia de seguridad).

## Configuración para Producción

Para usar HTTPS en producción con el dominio www.datainsight.com:

### Si usas Windows/IIS:

1. **Obtén certificados válidos con Let's Encrypt usando Win-ACME**:
   - Descarga Win-ACME desde https://github.com/win-acme/win-acme/releases
   - Extrae los archivos a una carpeta como `C:\win-acme`
   - Ejecuta `wacs.exe` y sigue el asistente para generar certificados para datainsight.com

2. **Configura el archivo .env para producción**:
   ```
   NODE_ENV=production
   SSL_ENABLED=true
   SSL_KEY_PATH=/etc/letsencrypt/live/datainsight.com/privkey.pem
   SSL_CERT_PATH=/etc/letsencrypt/live/datainsight.com/fullchain.pem
   ```

3. **Configura el binding HTTPS en IIS** con el certificado generado.

### Si usas otro proveedor de hosting:

Adapta las rutas de los certificados según corresponda para tu entorno de hosting.

## Solución de problemas comunes

### "No se puede acceder a https://localhost:45678" o "ERR_CONNECTION_REFUSED"

**Causas posibles y soluciones**:

1. **Puerto bloqueado/requiere privilegios**: 
   - Usa un puerto más alto (por defecto ahora usamos 9443)
   - O ejecuta la aplicación como administrador

2. **Certificados no reconocidos**:
   - Asegúrate de que los certificados existen en la carpeta `ssl/`
   - Acepta el certificado manualmente en el navegador

3. **Configuración de GnuTLS incorrecta**:
   - Verifica que GnuTLS se instaló correctamente: `gnutls-cli --version`
   - Reinstala siguiendo los pasos de la sección [Instalación de GnuTLS](#instalación-de-gnutls)

### "No se encuentra el archivo 'gnutls-cli'"

- Verifica que GnuTLS está en el PATH del sistema
- Reinicia tu terminal después de instalar GnuTLS
- Si sigue fallando, especifica la ruta completa al ejecutarlo

### "Error al iniciar servidor HTTPS"

- Verifica los logs para detalles del error
- Asegúrate de que los archivos de certificado existen y son accesibles
- Comprueba que no hay otro proceso usando el puerto HTTPS
