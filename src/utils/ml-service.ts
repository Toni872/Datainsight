import axios from 'axios';
import logger from './logger';

/**
 * Interfaz para las solicitudes de predicción
 */
interface PredictionRequest {
  features: number[];
  model_params?: Record<string, any>;
}

/**
 * Interfaz para las solicitudes de entrenamiento
 */
interface TrainModelRequest {
  dataset_id: string;
  model_type: string;
  model_name: string;
  test_size?: number;
  model_params?: Record<string, any>;
  optimize_hyperparams?: boolean;
}

/**
 * Interfaz para las solicitudes de clustering
 */
interface ClusteringRequest {
  dataset_id: string;
  algorithm: string;
  n_clusters?: number;
  model_params?: Record<string, any>;
  dimensionality_reduction?: string;
}

/**
 * Clase para integración con el servicio de Machine Learning en Azure
 * Esta clase se comunica con el servicio de ML desplegado en Azure Container Apps
 */
export class MLService {
  private mlServiceUrl: string;
  private maxRetries: number;
  private retryDelay: number;  private advancedApiAvailable: boolean = false;
  
  constructor() {
    // Obtiene la URL del servicio ML desde variables de entorno o usa una URL predeterminada
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    this.maxRetries = parseInt(process.env.ML_SERVICE_MAX_RETRIES || '3', 10);
    this.retryDelay = parseInt(process.env.ML_SERVICE_RETRY_DELAY_MS || '1000', 10);
    
    logger.info(`MLService initialized with URL: ${this.mlServiceUrl}`);
    
    // Verificar si la API avanzada está disponible
    this.checkAdvancedApiAvailability();
  }
  
  /**
   * Verifica si la API avanzada está disponible
   * @private
   */
  private async checkAdvancedApiAvailability(): Promise<void> {
    try {
      const response = await axios.get(`${this.mlServiceUrl}/`);
      const version = response.data?.version || '1.0.0';
      this.advancedApiAvailable = parseFloat(version) >= 2.0;
      logger.info(`ML API version detected: ${version}, advanced features ${this.advancedApiAvailable ? 'available' : 'not available'}`);
    } catch (error) {
      logger.warn('Could not determine ML API version, assuming basic API');
      this.advancedApiAvailable = false;
    }
  }
  
  /**
   * Verifica si la API avanzada está disponible
   */
  public isAdvancedApiAvailable(): boolean {
    return this.advancedApiAvailable;
  }
  
  /**
   * Obtiene el estado del servicio ML
   */
  public async getServiceStatus(): Promise<any> {
    try {
      // Intenta obtener el estado del servicio
      const response = await axios.get(`${this.mlServiceUrl}/status`, {
        timeout: 5000
      });
      
      // Si la respuesta incluye versión API v2, establecer modo avanzado
      if (response.data.version && response.data.version.includes('v2')) {
        this.advancedApiAvailable = true;
      } else {
        this.advancedApiAvailable = false;
      }
      
      logger.info(`ML Service status: ${JSON.stringify(response.data)}`);
      return response.data;
    } catch (error) {
      logger.error('Error al obtener estado del servicio ML:', error);
      this.advancedApiAvailable = false;
      throw error;
    }
  }
  
  /**
   * Realiza una petición HTTP con reintentos
   * @param method Método HTTP
   * @param endpoint Endpoint de la API
   * @param data Datos para enviar (opcional)
   * @returns Respuesta de la API
   */
  private async request<T>(method: string, endpoint: string, data?: any): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await axios({
          method,
          url: `${this.mlServiceUrl}${endpoint}`,
          data,
          timeout: 10000, // 10 segundos de timeout
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        return response.data;
      } catch (error: any) {
        lastError = error;
        const status = error.response?.status || 'desconocido';
        logger.warn(`Intento ${attempt + 1}/${this.maxRetries} fallido al ${method} ${endpoint}: ${error.message} (status: ${status})`);
        
        // Solo reintentar si es un error de red o un error 5xx
        if (error.response && error.response.status < 500 && error.response.status !== 0) {
          break;
        }
        
        // Si no es el último intento, esperar antes de reintentar
        if (attempt < this.maxRetries - 1) {
          // Retroceso exponencial
          const delayMs = this.retryDelay * Math.pow(2, attempt);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }
    
    throw lastError;
  }
  
  /**
   * Obtiene la lista de modelos disponibles en el servicio ML
   * @returns Información sobre modelos disponibles
   */
  public async getAvailableModels(): Promise<any> {
    try {
      // Si no está disponible la API avanzada, devolvemos los modelos básicos
      if (!this.advancedApiAvailable) {
        return {
          models: {
            classification: ['logistic_regression', 'decision_tree', 'random_forest', 'svm'],
            regression: ['linear_regression', 'decision_tree', 'random_forest', 'svr']
          }
        };
      }
      
      const response = await axios.get(`${this.mlServiceUrl}/models/available`, {
        timeout: 10000
      });
      
      logger.info(`Available models: ${JSON.stringify(response.data)}`);
      return response.data;
    } catch (error) {
      logger.error('Error al obtener modelos disponibles:', error);
      throw error;
    }
  }
  
  /**
   * Realiza una predicción utilizando un modelo específico
   * @param modelName Nombre del modelo a utilizar
   * @param features Características para la predicción
   * @param modelParams Parámetros adicionales para el modelo (opcional)
   * @returns Resultado de la predicción
   */
  async predict(modelName: string, features: number[], modelParams?: Record<string, any>): Promise<any> {
    try {
      const requestData: PredictionRequest = {
        features: features
      };
        if (modelParams && this.isAdvancedApiAvailable()) {
        requestData.model_params = modelParams;
      }
      
      return await this.request('post', `/predict/${modelName}`, requestData);
    } catch (error) {
      logger.error(`Error en predicción con modelo ${modelName}:`, error);
      throw new Error(`Error al realizar la predicción con el modelo ${modelName}`);
    }
  }
  
  /**
   * Entrena un modelo con los parámetros especificados
   * @param params Parámetros para entrenar el modelo
   * @returns Resultado del entrenamiento
   */
  public async trainModel(params: TrainModelRequest): Promise<any> {
    try {
      const response = await axios.post(`${this.mlServiceUrl}/models/train`, params, {
        timeout: 30000 // Los entrenamientos pueden llevar tiempo
      });
      
      logger.info(`Model training result: ${JSON.stringify(response.data)}`);
      return response.data;
    } catch (error) {
      logger.error('Error al entrenar modelo:', error);
      throw error;
    }
  }
  
  /**
   * Realiza clustering en un dataset
   * @param params Parámetros para clustering
   * @returns Resultado del clustering
   */
  public async performClustering(params: ClusteringRequest): Promise<any> {
    try {
      const response = await axios.post(`${this.mlServiceUrl}/clustering/perform`, params, {
        timeout: 20000
      });
      
      logger.info(`Clustering result: Status ${response.status}`);
      return response.data;
    } catch (error) {
      logger.error('Error al realizar clustering:', error);
      throw error;
    }
  }
  
  /**
   * Evalúa un modelo y genera visualizaciones
   * @param modelId ID del modelo a evaluar
   * @param evaluationType Tipo de evaluación a realizar
   * @param params Parámetros adicionales para la evaluación
   * @returns Resultado de la evaluación
   */
  public async evaluateModel(modelId: string, evaluationType: string, params?: Record<string, any>): Promise<any> {
    try {
      if (!this.isAdvancedApiAvailable()) {
        throw new Error('La funcionalidad de evaluación avanzada no está disponible');
      }
      
      const requestData = {
        model_id: modelId,
        evaluation_type: evaluationType,
        params: params || {}
      };
      
      return await this.request('post', `/models/${modelId}/evaluate`, requestData);
    } catch (error) {
      logger.error(`Error evaluando modelo ${modelId}:`, error);
      throw new Error(`Error al evaluar el modelo ${modelId}`);
    }
  }
  
  /**
   * Analiza una serie temporal
   * @param datasetId ID del dataset a analizar
   * @param analysisType Tipo de análisis a realizar
   * @param params Parámetros adicionales para el análisis
   * @returns Resultado del análisis
   */
  async analyzeTimeSeries(datasetId: string, analysisType: string, params?: Record<string, any>): Promise<any> {
    try {
      if (!this.isAdvancedApiAvailable()) {
        throw new Error('La funcionalidad de análisis de series temporales no está disponible');
      }
      
      const requestData = {
        dataset_id: datasetId,
        analysis_type: analysisType,
        params: params || {}
      };
      
      return await this.request('post', '/time-series/analyze', requestData);
    } catch (error) {
      logger.error(`Error analizando serie temporal ${datasetId}:`, error);
      throw new Error(`Error al analizar la serie temporal ${datasetId}`);
    }
  }
  
  /**
   * Predice valores futuros en una serie temporal
   * @param datasetId ID del dataset a analizar
   * @param modelName Nombre del modelo a utilizar
   * @param steps Número de pasos a predecir
   * @param params Parámetros adicionales para la predicción
   * @returns Resultado de la predicción
   */
  async forecastTimeSeries(datasetId: string, modelName: string, steps: number, params?: Record<string, any>): Promise<any> {
    try {
      if (!this.isAdvancedApiAvailable()) {
        throw new Error('La funcionalidad de predicción de series temporales no está disponible');
      }
      
      const requestData = {
        dataset_id: datasetId,
        model_name: modelName,
        steps: steps,
        params: params || {}
      };
      
      return await this.request('post', '/time-series/forecast', requestData);
    } catch (error) {
      logger.error(`Error prediciendo serie temporal ${datasetId}:`, error);
      throw new Error(`Error al predecir la serie temporal ${datasetId}`);
    }
  }
  
  /**
   * Obtiene la lista de datasets disponibles
   * @returns Lista de datasets disponibles
   */
  async getAvailableDatasets(): Promise<string[]> {
    try {
      const data = await this.request<{ available_datasets: string[] }>('get', '/datasets');
      return data.available_datasets || [];
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
      await this.request('get', '/');
      return true;
    } catch (error) {
      logger.warn('No se pudo conectar con el servicio ML:', error);
      return false;
    }
  }
}

// Exportar una instancia de la clase para usar en toda la aplicación
export const mlService = new MLService();