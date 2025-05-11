/**
 * Sistema de registro (logger) para la aplicación
 */
import { createLogger, format, transports } from 'winston';

/**
 * Niveles de registro disponibles
 */
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Logger utilizando winston para la aplicación
 */
const logger = createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: format.combine(
    format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    format.printf(info => `[${info.timestamp}] [${info.level.toUpperCase()}] ${info.message}`)
  ),
  transports: [
    new transports.Console()
  ]
});

/**
 * Si estamos en ambiente de producción, también escribimos a un archivo
 */
if (process.env.NODE_ENV === 'production') {
  logger.add(new transports.File({ filename: 'logs/error.log', level: 'error' }));
  logger.add(new transports.File({ filename: 'logs/combined.log' }));
}

export default logger;