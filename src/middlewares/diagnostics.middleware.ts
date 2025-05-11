import { Request, Response, NextFunction } from 'express';
import logger from '../utils/logger';

/**
 * Middleware para registrar información detallada de solicitudes HTTP
 * y diagnosticar problemas potenciales de rendimiento y errores
 */
export const diagnosticsMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  // Registrar inicio de la solicitud
  const startTime = Date.now();
  const requestId = `req-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
  
  logger.info(`[${requestId}] Solicitud iniciada: ${req.method} ${req.originalUrl}`);
  
  // Capturar detalles de la solicitud para diagnóstico
  const requestDetails = {
    method: req.method,
    url: req.originalUrl,
    headers: req.headers,
    query: req.query,
    body: req.method !== 'GET' ? req.body : undefined,
    ip: req.ip,
    timestamp: new Date().toISOString()
  };
  
  logger.debug(`[${requestId}] Detalles de solicitud: ${JSON.stringify(requestDetails)}`);

  // Capturar y monitorear la respuesta
  const originalSend = res.send;
  res.send = function(body) {
    const responseTime = Date.now() - startTime;
    const responseSize = body ? (typeof body === 'string' ? body.length : JSON.stringify(body).length) : 0;
    
    // Registrar métricas de respuesta
    logger.info(`[${requestId}] Respuesta completada: ${res.statusCode} en ${responseTime}ms (tamaño: ${responseSize} bytes)`);
    
    // Registrar tiempos de respuesta lentos (más de 1 segundo)
    if (responseTime > 1000) {
      logger.warn(`[${requestId}] Respuesta lenta detectada: ${responseTime}ms para ${req.method} ${req.originalUrl}`);
    }
    
    // Registrar errores 4xx y 5xx con detalles adicionales
    if (res.statusCode >= 400) {
      const errorLevel = res.statusCode >= 500 ? 'error' : 'warn';
      logger[errorLevel](`[${requestId}] Error de respuesta: ${res.statusCode} para ${req.method} ${req.originalUrl}`);
      
      // Para errores 5xx, registrar detalles adicionales para diagnóstico
      if (res.statusCode >= 500) {
        logger.error(`[${requestId}] Detalles de error 5xx: 
          - URL: ${req.originalUrl}
          - Método: ${req.method}
          - Tiempo de respuesta: ${responseTime}ms
          - Encabezados: ${JSON.stringify(req.headers)}
          - Respuesta: ${typeof body === 'string' ? body.substring(0, 500) : JSON.stringify(body).substring(0, 500)}`);
      }
    }
    
    return originalSend.call(this, body);
  };
  
  // Manejo de cierre inesperado de la conexión
  req.on('close', () => {
    if (!res.writableEnded) {
      logger.warn(`[${requestId}] Conexión cerrada antes de completar la respuesta para ${req.method} ${req.originalUrl}`);
    }
  });
  
  next();
};

/**
 * Middleware para manejo global de errores con diagnóstico detallado
 */
export const enhancedErrorHandler = (err: any, req: Request, res: Response, next: NextFunction): void => {
  const errorId = `err-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
  
  // Registrar el error con detalles
  logger.error(`[${errorId}] Error no controlado: ${err.message}`);
  logger.error(`[${errorId}] Stack trace: ${err.stack}`);
  
  // Registrar contexto de la solicitud
  logger.error(`[${errorId}] Contexto de la solicitud que causó el error:
    - URL: ${req.originalUrl}
    - Método: ${req.method}
    - IP: ${req.ip}
    - Usuario-Agente: ${req.headers['user-agent']}
    - Referrer: ${req.headers.referer || 'No referrer'}`
  );
  
  // Devolver respuesta de error estructurada
  res.status(500).json({
    status: 'error',
    message: 'Se ha producido un error en el servidor',
    errorId: errorId, // Proporcionar un ID de error para referencia
    requestId: req.headers['x-request-id'] || 'unknown'
  });
};

/**
 * Middleware para verificar la salud del sistema y sus dependencias
 */
export const healthCheckMiddleware = async (req: Request, res: Response): Promise<void> => {
  try {
    // Verificar recursos del sistema
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    // Verificar variables de entorno críticas
    const environmentCheck = {
      node_env: process.env.NODE_ENV || 'not set',
      port: process.env.PORT || '3000',
      python_path: process.env.PYTHON_PATH || 'default',
      ml_service_url: process.env.ML_SERVICE_URL || 'not set'
    };
    
    // Verificar espacio en disco (simplificado)
    let diskSpace = 'No disponible';
    try {
      const fs = require('fs');
      const os = require('os');
      const tempDir = os.tmpdir();
      fs.accessSync(tempDir, fs.constants.W_OK);
      diskSpace = 'Espacio de escritura disponible';
    } catch (e) {
      diskSpace = `Error al verificar espacio en disco: ${e instanceof Error ? e.message : 'Error desconocido'}`;
    }
    
    res.status(200).json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memoryUsage: {
        rss: `${Math.round(memoryUsage.rss / 1024 / 1024)} MB`,
        heapTotal: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)} MB`,
        heapUsed: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)} MB`,
        external: `${Math.round(memoryUsage.external / 1024 / 1024)} MB`,
      },
      cpuUsage: {
        user: cpuUsage.user,
        system: cpuUsage.system
      },
      environment: environmentCheck,
      diskSpace: diskSpace
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: `Error al verificar la salud del sistema: ${error instanceof Error ? error.message : 'Error desconocido'}`
    });
  }
};