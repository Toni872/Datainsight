import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

// Exportamos todas las utilidades
export * from './common';
export * from './python-integration';

/**
 * Constantes globales de la aplicación
 */
export const APP_CONFIG = {
  APP_NAME: 'Mi Proyecto',
  VERSION: '1.0.0',
  DEFAULT_LANGUAGE: 'es-ES',
  MAX_FILE_SIZE_MB: 10,
  API_TIMEOUT_MS: 30000,
};

/**
 * Funciones básicas de logging
 */
export const logger = (message: string): void => {
  console.log(`[${new Date().toISOString()}] INFO: ${message}`);
};

export const errorLogger = (error: Error): void => {
  console.error(`[${new Date().toISOString()}] ERROR: ${error.message}`);
  console.error(error.stack);
};