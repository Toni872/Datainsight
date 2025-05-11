import { Request, Response } from 'express';
import * as path from 'path';
import * as fs from 'fs';
import { PythonIntegration } from '../utils/python-integration';
import { MLService } from '../utils/ml-service';
import logger from '../utils/logger';
import * as multer from 'multer';

// Definir un tipo extendido de Request que incluya la propiedad file
interface MulterRequest extends Request {
  file?: Express.Multer.File;
}

/**
 * Controlador para los endpoints de API relacionados con modelos de machine learning
 */
export class ModelsController {
  private pythonIntegration: PythonIntegration;
  private mlService: MLService;
  private tempDir: string;

  constructor() {
    this.pythonIntegration = new PythonIntegration();
    this.mlService = new MLService();
    
    // Crear directorio temporal para datos intermedios si no existe
    this.tempDir = path.join(__dirname, '../../temp');
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  /**
   * Lista todos los datasets disponibles
   */
  public listDatasets = async (req: Request, res: Response): Promise<void> => {
    try {
      const scriptPath = '3_machine_learning/supervisado/dataset_loader.py';
      const output = await this.pythonIntegration.executeScript(scriptPath, ['list']);
      
      const datasets = JSON.parse(output);
      res.json({ success: true, datasets });
    } catch (error) {
      logger.error('Error al listar datasets:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };

  /**
   * Obtiene información detallada de un dataset específico
   */
  public getDatasetInfo = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      
      if (!id) {
        res.status(400).json({ success: false, error: 'Se requiere un ID de dataset' });
        return;
      }
      
      const scriptPath = '3_machine_learning/supervisado/dataset_loader.py';
      const output = await this.pythonIntegration.executeScript(scriptPath, ['get_info', id]);
      
      const info = JSON.parse(output);
      
      if (info.error) {
        res.status(404).json({ success: false, error: info.error });
      } else {
        res.json({ success: true, ...info });
      }
    } catch (error) {
      logger.error(`Error al obtener información del dataset ${req.params.id}:`, error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };

  /**
   * Guarda un nuevo dataset subido por el usuario
   */
  public uploadDataset = async (req: MulterRequest, res: Response): Promise<void> => {
    try {
      if (!req.file) {
        res.status(400).json({ success: false, error: 'No se ha subido ningún archivo' });
        return;
      }
      
      const { name, target_column, description } = req.body;
      
      if (!name) {
        res.status(400).json({ success: false, error: 'Se requiere un nombre para el dataset' });
        return;
      }
      
      // Ruta del archivo subido
      const filePath = req.file.path;
      
      const scriptPath = '3_machine_learning/supervisado/dataset_loader.py';
      const args = ['save', filePath, name];
      
      // Añadir parámetros opcionales si se proporcionan
      if (target_column) args.push(target_column);
      if (description) args.push(description);
      
      const output = await this.pythonIntegration.executeScript(scriptPath, args);
      const result = JSON.parse(output);
      
      if (result.success) {
        res.json({ success: true, dataset: result });
      } else {
        res.status(500).json({ success: false, error: result.error });
      }
      
      // Eliminar el archivo temporal después de procesarlo
      fs.unlinkSync(filePath);
      
    } catch (error) {
      logger.error('Error al subir dataset:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
      
      // Si hay un archivo subido, asegurarse de eliminarlo
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
    }
  };

  /**
   * Entrena modelos de ML con el dataset especificado
   */
  public trainModels = async (req: Request, res: Response): Promise<void> => {
    try {
      const { datasetId, modelType, models, testSize, cvFolds, scaling } = req.body;
      
      if (!datasetId || !modelType || !models || !Array.isArray(models) || models.length === 0) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requiere datasetId, modelType y un array de modelos' 
        });
        return;
      }
      
      // Paso 1: Preparar los datos
      const datasetLoaderPath = '3_machine_learning/supervisado/dataset_loader.py';
      const datasetArgs = [
        'prepare_data', 
        datasetId, 
        testSize ? testSize.toString() : '0.2', 
        scaling !== undefined ? scaling.toString() : 'true'
      ];
      
      const prepOutput = await this.pythonIntegration.executeScript(datasetLoaderPath, datasetArgs);
      const prepResult = JSON.parse(prepOutput);
      
      if (!prepResult.success || !prepResult.temp_file) {
        res.status(500).json({ success: false, error: 'Error al preparar los datos' });
        return;
      }
      
      // Paso 2: Entrenar los modelos
      const modelTrainerPath = '3_machine_learning/supervisado/model_trainer.py';
      const trainArgs = [
        'train', 
        prepResult.temp_file, 
        modelType, 
        models.join(','), 
        cvFolds ? cvFolds.toString() : '5'
      ];
      
      const trainOutput = await this.pythonIntegration.executeScript(modelTrainerPath, trainArgs);
      const trainResult = JSON.parse(trainOutput);
      
      if (trainResult.error) {
        res.status(500).json({ success: false, error: trainResult.error });
      } else {
        res.json({ success: true, results: trainResult });
      }
      
      // Eliminar archivo temporal de datos
      if (fs.existsSync(prepResult.temp_file)) {
        fs.unlinkSync(prepResult.temp_file);
      }
      
    } catch (error) {
      logger.error('Error al entrenar modelos:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };

  /**
   * Realiza predicciones con un modelo entrenado
   */
  public predict = async (req: Request, res: Response): Promise<void> => {
    try {
      const { modelName, datasetId, features } = req.body;
      
      if (!modelName || !datasetId || !features || !Array.isArray(features)) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requieren modelName, datasetId y un array de características' 
        });
        return;
      }
      
      const scriptPath = '3_machine_learning/supervisado/model_trainer.py';
      const args = ['predict', modelName, datasetId, JSON.stringify(features)];
      
      const output = await this.pythonIntegration.executeScript(scriptPath, args);
      const result = JSON.parse(output);
      
      if (result.error) {
        res.status(500).json({ success: false, error: result.error });
      } else {
        res.json({ success: true, prediction: result });
      }
      
    } catch (error) {
      logger.error('Error al realizar predicción:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };

  /**
   * Obtiene el estado del servicio ML y verifica disponibilidad de modelos avanzados
   */
  public getMLServiceStatus = async (req: Request, res: Response): Promise<void> => {
    try {
      const status = await this.mlService.getServiceStatus();
      res.json({
        success: true,
        ...status,
        isAdvancedApiAvailable: this.mlService.isAdvancedApiAvailable()
      });
    } catch (error) {
      logger.error('Error al obtener estado del servicio ML:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Error desconocido',
        isAdvancedApiAvailable: false
      });
    }
  };
  
  /**
   * Obtiene la lista de modelos disponibles y sus capacidades
   */
  public getAvailableModels = async (req: Request, res: Response): Promise<void> => {
    try {
      const models = await this.mlService.getAvailableModels();
      res.json({
        success: true,
        ...models
      });
    } catch (error) {
      logger.error('Error al obtener modelos disponibles:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Error desconocido'
      });
    }
  };
  
  /**
   * Entrena un modelo avanzado
   */
  public trainAdvancedModel = async (req: Request, res: Response): Promise<void> => {
    try {
      const { dataset_id, model_type, model_name, test_size, model_params, optimize_hyperparams } = req.body;
      
      if (!dataset_id || !model_type || !model_name) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requieren dataset_id, model_type y model_name'
        });
        return;
      }
      
      // Verificar si la API avanzada está disponible
      if (!this.mlService.isAdvancedApiAvailable()) {
        // Usar el script Python convencional si no está disponible
        logger.info('API avanzada no disponible, usando script Python convencional');
        
        const scriptPath = '3_machine_learning/supervisado/model_trainer.py';
        const args = ['train', dataset_id, model_type, model_name];
        
        // Añadir parámetros opcionales
        if (test_size) args.push('--test_size', test_size.toString());
        if (optimize_hyperparams) args.push('--optimize');
        
        // Añadir parámetros del modelo si existen
        if (model_params) {
          args.push('--params', JSON.stringify(model_params));
        }
        
        const output = await this.pythonIntegration.executeScript(scriptPath, args);
        const result = JSON.parse(output);
        
        if (result.error) {
          res.status(400).json({ success: false, error: result.error });
        } else {
          res.json({ success: true, ...result });
        }
      } else {
        // Usar la API avanzada
        logger.info('Usando API avanzada para entrenar modelo');
        const result = await this.mlService.trainModel({
          dataset_id,
          model_type,
          model_name,
          test_size: test_size || 0.2,
          model_params,
          optimize_hyperparams: optimize_hyperparams || false
        });
        
        res.json({ success: true, ...result });
      }
    } catch (error) {
      logger.error('Error al entrenar modelo avanzado:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };
  
  /**
   * Realiza clustering en un dataset
   */
  public performClustering = async (req: Request, res: Response): Promise<void> => {
    try {
      const { dataset_id, algorithm, n_clusters, model_params, dimensionality_reduction } = req.body;
      
      if (!dataset_id || !algorithm) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requieren dataset_id y algorithm' 
        });
        return;
      }
      
      if (!this.mlService.isAdvancedApiAvailable()) {
        res.status(400).json({
          success: false,
          error: 'La funcionalidad de clustering avanzado no está disponible'
        });
        return;
      }
      
      const result = await this.mlService.performClustering({
        dataset_id,
        algorithm,
        n_clusters,
        model_params,
        dimensionality_reduction
      });
      
      res.json({ success: true, ...result });
    } catch (error) {
      logger.error('Error al realizar clustering:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };
  
  /**
   * Realiza análisis de series temporales
   */
  public analyzeTimeSeries = async (req: Request, res: Response): Promise<void> => {
    try {
      const { dataset_id, analysis_type, params } = req.body;
      
      if (!dataset_id || !analysis_type) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requieren dataset_id y analysis_type' 
        });
        return;
      }
      
      // Usar el script Python directamente, ya que esto podría no estar en la API
      const scriptPath = '5_especializacion/series_temporales/time_series_analyzer.py';
      const args = [analysis_type, dataset_id];
      
      if (params) {
        args.push('--params', JSON.stringify(params));
      }
      
      const output = await this.pythonIntegration.executeScript(scriptPath, args);
      const result = JSON.parse(output);
      
      if (result.error) {
        res.status(400).json({ success: false, error: result.error });
      } else {
        res.json({ success: true, ...result });
      }
    } catch (error) {
      logger.error('Error al analizar series temporales:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };
  
  /**
   * Genera visualizaciones avanzadas para la evaluación de modelos
   */
  public generateModelVisualization = async (req: Request, res: Response): Promise<void> => {
    try {
      const { model_id, visualization_type, params } = req.body;
      
      if (!model_id || !visualization_type) {
        res.status(400).json({ 
          success: false, 
          error: 'Se requieren model_id y visualization_type' 
        });
        return;
      }
      
      if (!this.mlService.isAdvancedApiAvailable()) {
        // Usar script tradicional para visualizaciones básicas
        const scriptPath = '3_machine_learning/supervisado/model_evaluator.py';
        const args = ['visualize', model_id, visualization_type];
        
        if (params) {
          args.push('--params', JSON.stringify(params));
        }
        
        const output = await this.pythonIntegration.executeScript(scriptPath, args);
        const result = JSON.parse(output);
        
        if (result.error) {
          res.status(400).json({ success: false, error: result.error });
        } else {
          res.json({ success: true, ...result });
        }
      } else {
        // Usar API avanzada
        const result = await this.mlService.evaluateModel(
          model_id, 
          visualization_type, 
          params
        );
        
        res.json({ success: true, ...result });
      }
    } catch (error) {
      logger.error('Error al generar visualización del modelo:', error);
      res.status(500).json({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Error desconocido' 
      });
    }
  };
}