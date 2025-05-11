import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import axios from 'axios';
import logger from './logger';

/**
 * Clase para manejar la integración con scripts de Python
 */
export class PythonIntegration {
  private pythonPath: string;
  private scriptsBasePath: string;
  private retryAttempts: number;
  private retryDelay: number;

  constructor() {
    // En producción, es mejor configurar estas rutas mediante variables de entorno
    this.pythonPath = process.env.PYTHON_PATH || 'python';
    this.scriptsBasePath = process.env.PYTHON_SCRIPTS_PATH || 
                         path.join(__dirname, '../../data_science_ml_learning');
    this.retryAttempts = parseInt(process.env.PYTHON_RETRY_ATTEMPTS || '3');
    this.retryDelay = parseInt(process.env.PYTHON_RETRY_DELAY_MS || '1000');
    
    logger.info(`PythonIntegration initialized with scripts path: ${this.scriptsBasePath}`);
  }

  /**
   * Ejecuta un script de Python y devuelve la salida como una promesa
   * @param scriptPath Ruta relativa al script dentro de la carpeta data_science_ml_learning
   * @param args Argumentos para pasar al script
   * @returns Promise con la salida del script
   */
  async executeScript(scriptPath: string, args: string[] = []): Promise<string> {
    const fullPath = path.join(this.scriptsBasePath, scriptPath);
    
    // Verifica que el archivo exista
    if (!fs.existsSync(fullPath)) {
      logger.error(`Script no encontrado: ${fullPath}`);
      throw new Error(`Script no encontrado: ${fullPath}`);
    }

    logger.debug(`Ejecutando script Python: ${fullPath} con argumentos: ${args.join(', ')}`);
    
    return this.executeWithRetry(async () => {
      return new Promise((resolve, reject) => {
        // Especificamos UTF-8 como codificación para el proceso
        const pythonProcess = spawn(this.pythonPath, [fullPath, ...args], {
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8'  // Forzar codificación UTF-8 para entrada/salida
          }
        });
        
        let outputData = '';
        let errorData = '';
        
        // Configurar explícitamente la codificación UTF-8 para los streams
        pythonProcess.stdout.setEncoding('utf-8');
        pythonProcess.stderr.setEncoding('utf-8');
        
        pythonProcess.stdout.on('data', (data) => {
          outputData += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          errorData += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            logger.error(`Error en script Python (código ${code}): ${errorData}`);
            reject(new Error(`Error al ejecutar script Python: ${errorData}`));
          } else {
            logger.debug(`Script Python ejecutado exitosamente`);
            resolve(outputData);
          }
        });

        // Manejo de errores en el proceso
        pythonProcess.on('error', (err) => {
          logger.error(`Error al iniciar proceso Python: ${err.message}`);
          reject(new Error(`Error al iniciar proceso Python: ${err.message}`));
        });
        
        // Timeout para evitar procesos colgados
        const timeout = setTimeout(() => {
          pythonProcess.kill();
          logger.error(`Timeout al ejecutar script Python: ${fullPath}`);
          reject(new Error('Timeout al ejecutar script Python'));
        }, 30000); // 30 segundos de timeout
        
        // Limpiar el timeout cuando el proceso termina
        pythonProcess.on('close', () => clearTimeout(timeout));
      });
    });
  }

  /**
   * Ejecuta una función con lógica de reintentos con retroceso exponencial
   * @param fn Función a ejecutar
   * @returns Resultado de la función
   */
  private async executeWithRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        logger.warn(`Intento ${attempt + 1}/${this.retryAttempts} fallido: ${lastError.message}`);
        
        // Si no es el último intento, esperar antes de reintentar
        if (attempt < this.retryAttempts - 1) {
          // Retroceso exponencial: tiempo base * 2^intentos
          const delayMs = this.retryDelay * Math.pow(2, attempt);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }
    
    throw lastError;
  }

  /**
   * Verifica la integración con Python ejecutando el script de prueba
   * @returns Información del entorno Python
   */
  async testIntegration(): Promise<any> {
    try {
      const output = await this.executeScript('integration_test.py', ['--test-mode']);
      return JSON.parse(output);
    } catch (error) {
      logger.error('Error en la prueba de integración Python:', error);
      throw new Error('Error al verificar la integración con Python');
    }
  }
  
  /**
   * Entrena un modelo de machine learning usando los scripts Python
   * @param modelType Tipo de modelo (ej: 'random_forest', 'svm')
   * @param datasetName Nombre del dataset a utilizar
   * @param params Parámetros adicionales para el entrenamiento
   * @returns Detalles del modelo entrenado
   */
  async trainModel(modelType: string, datasetName: string, params: Record<string, any> = {}): Promise<any> {
    try {
      // Convertir parámetros a formato que Python pueda parsear
      const paramArgs = Object.entries(params).map(([key, value]) => `--${key}=${value}`);
      
      const output = await this.executeScript(
        path.join('3_machine_learning', 'supervisado', 'model_trainer.py'),
        [modelType, datasetName, ...paramArgs]
      );
      
      return JSON.parse(output);
    } catch (error) {
      logger.error(`Error al entrenar modelo ${modelType} con dataset ${datasetName}:`, error);
      throw new Error(`Error al entrenar el modelo ${modelType}`);
    }
  }
}

/**
 * Clase para integración con el servicio de Machine Learning en Azure
 */
export class MLService {
  private mlServiceUrl: string;
  
  constructor() {
    // Obtiene la URL del servicio ML desde variables de entorno o usa una URL predeterminada
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    logger.info(`MLService initialized with URL: ${this.mlServiceUrl}`);
  }
  
  /**
   * Obtiene la lista de modelos disponibles en el servicio ML
   * @returns Lista de modelos disponibles
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await axios.get(`${this.mlServiceUrl}/models`);
      return response.data.available_models || [];
    } catch (error) {
      logger.error('Error obteniendo modelos disponibles:', error);
      throw new Error('Error al obtener los modelos disponibles');
    }
  }
  
  /**
   * Realiza una predicción utilizando un modelo específico
   * @param modelName Nombre del modelo a utilizar
   * @param features Características para la predicción
   * @returns Resultado de la predicción
   */
  async predict(modelName: string, features: number[]): Promise<any> {
    try {
      const response = await axios.post(`${this.mlServiceUrl}/predict/${modelName}`, {
        features: features
      });
      return response.data;
    } catch (error) {
      logger.error(`Error en predicción con modelo ${modelName}:`, error);
      throw new Error(`Error al realizar la predicción con el modelo ${modelName}`);
    }
  }
  
  /**
   * Obtiene la lista de datasets disponibles
   * @returns Lista de datasets disponibles
   */
  async getAvailableDatasets(): Promise<string[]> {
    try {
      const response = await axios.get(`${this.mlServiceUrl}/datasets`);
      return response.data.available_datasets || [];
    } catch (error) {
      logger.error('Error obteniendo datasets disponibles:', error);
      throw new Error('Error al obtener los datasets disponibles');
    }
  }
  
  /**
   * Verifica la conectividad con el servicio ML
   * @returns true si el servicio está disponible, false en caso contrario
   */
  async checkConnectivity(): Promise<boolean> {
    try {
      await axios.get(this.mlServiceUrl);
      return true;
    } catch (error) {
      logger.warn('No se pudo conectar con el servicio ML:', error);
      return false;
    }
  }
}

// Reexportar la instancia de MLService para mantener la compatibilidad con el código existente
// export { mlService } from './ml-service'; // Removed due to missing module