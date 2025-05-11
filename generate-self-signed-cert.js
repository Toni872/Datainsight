// Script para generar certificados SSL autofirmados para desarrollo
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuración
const sslDir = path.join(__dirname, 'ssl');
const domain = 'localhost';
const keyPath = path.join(sslDir, 'privkey.pem');
const certPath = path.join(sslDir, 'fullchain.pem');

console.log('Generando certificados SSL autofirmados para desarrollo local...');

try {
  // Asegurarse de que el directorio ssl existe
  if (!fs.existsSync(sslDir)) {
    fs.mkdirSync(sslDir, { recursive: true });
    console.log(`Directorio ${sslDir} creado.`);
  }

  // Verificar si los certificados ya existen
  if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
    console.log('Los certificados ya existen. Si desea generarlos nuevamente, elimine los archivos existentes primero.');
    process.exit(0);
  }

  // Generar certificado autofirmado con OpenSSL
  // Nota: Esto requiere que OpenSSL esté instalado en el sistema
  const opensslCommand = `openssl req -x509 -newkey rsa:4096 -keyout "${keyPath}" -out "${certPath}" -days 365 -nodes -subj "/CN=${domain}"`;
  
  console.log('Ejecutando comando OpenSSL...');
  console.log(opensslCommand);
  
  execSync(opensslCommand, { stdio: 'inherit' });
  
  console.log('\nCertificados SSL autofirmados generados con éxito.');
  console.log('- Clave privada: ' + keyPath);
  console.log('- Certificado: ' + certPath);
  console.log('\nIMPORTANTE: Estos certificados son solo para desarrollo local.');
  console.log('Para producción, use certificados válidos de Let\'s Encrypt u otra CA.');

} catch (error) {
  console.error('Error al generar certificados SSL:', error.message);
  console.log('\nSi OpenSSL no está instalado, puede descargarlo desde:');
  console.log('https://slproweb.com/products/Win32OpenSSL.html');
  process.exit(1);
}
