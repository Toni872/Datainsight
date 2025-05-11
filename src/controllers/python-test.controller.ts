import { Request, Response } from 'express';
import { PythonIntegration } from '../utils/python-integration';
import { mlService } from '../utils/ml-service';
import logger from '../utils/logger';

/**
 * Controlador para probar la integración con Python y el servicio ML
 */
export class PythonTestController {
  private pythonIntegration: PythonIntegration;
  
  constructor() {
    this.pythonIntegration = new PythonIntegration();
  }

  /**
   * Ejecuta una prueba de integración con Python local
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  async testLocalPythonIntegration(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Iniciando prueba de integración con Python local');
      const result = await this.pythonIntegration.testIntegration();
      res.status(200).json({
        success: true,
        message: 'Integración con Python funcionando correctamente',
        pythonInfo: result
      });
    } catch (error: any) {
      logger.error('Error en prueba de integración con Python local:', error);
      res.status(500).json({
        success: false,
        message: 'Error en la integración con Python',
        error: error.message
      });
    }
  }

  /**
   * Prueba la conectividad con el servicio ML en Azure
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  async testAzureMLService(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Probando conectividad con servicio ML en Azure');
      const isConnected = await mlService.checkConnectivity();
      
      if (isConnected) {
        // Si hay conectividad, obtener modelos y datasets disponibles
        const [models, datasets] = await Promise.all([
          mlService.getAvailableModels(),
          mlService.getAvailableDatasets()
        ]);
        
        res.status(200).json({
          success: true,
          message: 'Conexión exitosa con el servicio ML',
          serviceInfo: {
            url: process.env.ML_SERVICE_URL || 'http://localhost:8000',
            availableModels: models,
            availableDatasets: datasets
          }
        });
      } else {
        res.status(200).json({
          success: false,
          message: 'No se pudo conectar con el servicio ML',
          serviceInfo: {
            url: process.env.ML_SERVICE_URL || 'http://localhost:8000'
          }
        });
      }
    } catch (error: any) {
      logger.error('Error probando conexión con servicio ML:', error);
      res.status(500).json({
        success: false,
        message: 'Error al probar conexión con servicio ML',
        error: error.message
      });
    }
  }

  /**
   * Controlador para realizar una predicción simple con un modelo de ejemplo (Iris)
   * @param req Solicitud HTTP
   * @param res Respuesta HTTP
   */
  async testSimplePrediction(req: Request, res: Response): Promise<Response> {
    try {
      // Estos son valores de ejemplo para el dataset de Iris (4 características)
      const features = [5.1, 3.5, 1.4, 0.2]; // Ejemplo de Iris-setosa
      
      logger.info(`Realizando predicción de prueba con features: ${features.join(', ')}`);
      
      // Intentar usar MLService primero (API remota)
      try {
        const mlResult = await mlService.predict('iris_classifier', features);
        return res.status(200).json({
          success: true,
          message: 'Predicción realizada con éxito usando el servicio ML',
          prediction: mlResult
        });
      } catch (mlError) {
        logger.warn('No se pudo usar MLService, intentando con integración Python local:', mlError);
        
        // Fallback a Python local
        const pythonResult = await this.pythonIntegration.executeScript(
          '3_machine_learning/supervisado/clasificacion_basica.py',
          ['--predict', ...features.map(String)]
        );
        
        return res.status(200).json({
          success: true,
          message: 'Predicción realizada con éxito usando Python local',
          prediction: JSON.parse(pythonResult)
        });
      }
    } catch (error: any) {
      logger.error('Error al realizar predicción de prueba:', error);
      return res.status(500).json({
        success: false,
        message: 'Error al realizar predicción de prueba',
        error: error.message
      });
    }
  }
}

export const pythonTestController = new PythonTestController();