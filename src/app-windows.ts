/**
 * Modificación especial para app.ts para solucionar problemas en Windows
 * Este archivo debe ser aplicado temporalmente para el desarrollo HTTPS
 */

import express from 'express';
import path from 'path';
import { setRoutes } from './routes';
import { errorHandler } from './middlewares';
import { diagnosticsMiddleware, enhancedErrorHandler } from './middlewares/diagnostics.middleware';
import https from 'https';
import fs from 'fs';
import dotenv from 'dotenv';
import { createServer } from 'http';

// Cargar variables de entorno desde el archivo .env
dotenv.config();

const app = express();

// Configuración de variables de entorno
const PORT = process.env.PORT || 3000;
const HTTPS_PORT = process.env.HTTPS_PORT || 59999;
const isDev = process.env.NODE_ENV !== 'production';
const DOMAIN = isDev ? 'localhost' : 'www.datainsight.com';
const ROOT_DOMAIN = isDev ? 'localhost' : 'datainsight.com';
const SSL_ENABLED = process.env.SSL_ENABLED === 'true' || false;

// Middleware para diagnóstico
app.use(diagnosticsMiddleware);

// Middleware configuration
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS configuration for domain
app.use((req, res, next) => {
  const allowedOrigins = [
    `http://${DOMAIN}:${PORT}`,
    `https://${DOMAIN}:${HTTPS_PORT}`,
    `http://${ROOT_DOMAIN}:${PORT}`,
    `https://${ROOT_DOMAIN}:${HTTPS_PORT}`,
    'http://localhost:3000',
    'https://localhost:59999'
  ];
  
  const origin = req.headers.origin;
  if (origin && allowedOrigins.includes(origin)) {
    res.header('Access-Control-Allow-Origin', origin);
  }
  
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  next();
});

// Servir archivos estáticos desde la carpeta 'public'
app.use(express.static(path.join(__dirname, '../public')));

// Set up routes
setRoutes(app);

// Error handling middleware
app.use(errorHandler);
app.use(enhancedErrorHandler);

// INICIO DE LA MODIFICACIÓN PARA WINDOWS

// Función para iniciar el servidor HTTP
function startHttpServer() {
  const httpServer = createServer(app);
  httpServer.listen(PORT, () => {
    console.log(`HTTP Server running on port ${PORT}`);
    console.log(`Sitio web disponible en: http://${isDev ? 'localhost' : DOMAIN}:${isDev ? PORT : ''}`);
  });
  return httpServer;
}

// Función para iniciar el servidor HTTPS
function startHttpsServer() {
  try {
    // Rutas a los archivos de certificado
    const privateKeyPath = isDev 
      ? path.join(__dirname, '../ssl/privkey.pem') 
      : process.env.SSL_KEY_PATH || '/etc/letsencrypt/live/datainsight.com/privkey.pem';
    
    const certificatePath = isDev
      ? path.join(__dirname, '../ssl/fullchain.pem')
      : process.env.SSL_CERT_PATH || '/etc/letsencrypt/live/datainsight.com/fullchain.pem';
    
    // Verificar que los archivos existen
    if (fs.existsSync(privateKeyPath) && fs.existsSync(certificatePath)) {
      // Configurar opciones SSL
      const httpsOptions = {
        key: fs.readFileSync(privateKeyPath),
        cert: fs.readFileSync(certificatePath)
      };
        // Crear servidor HTTPS
      const httpsServer = https.createServer(httpsOptions, app);
      httpsServer.on('error', (e: NodeJS.ErrnoException) => {
        console.error('Error al iniciar servidor HTTPS:', e.message);
        if (e.code === 'EACCES') {
          console.log('\nERROR DE PERMISOS: No se puede usar el puerto HTTPS.');
          console.log('SOLUCIÓN ALTERNATIVA: Prueba a usar el puerto HTTP en su lugar:');
          console.log(`http://localhost:${PORT}`);
        }
      });
      
      httpsServer.listen(HTTPS_PORT, () => {
        console.log(`HTTPS Server running on port ${HTTPS_PORT}`);
        console.log(`Sitio web seguro disponible en: https://${isDev ? 'localhost' : DOMAIN}:${isDev ? HTTPS_PORT : ''}`);
      });
      
      return httpsServer;
    } else {
      console.warn(`Archivos de certificado SSL no encontrados. Iniciando solo servidor HTTP.`);
      return null;
    }
  } catch (error) {
    console.error('Error al iniciar servidor HTTPS:', error);
    return null;
  }
}

// Iniciar los servidores
const httpServer = startHttpServer();
if (SSL_ENABLED) {
  const httpsServer = startHttpsServer();
  if (!httpsServer) {
    console.log('El servidor HTTPS no pudo iniciarse. Solo está disponible el servidor HTTP.');
  }
}

// FIN DE LA MODIFICACIÓN PARA WINDOWS

// Exportar la app para testing
export default app;
