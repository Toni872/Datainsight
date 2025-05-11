import { Request, Response } from 'express';
import { DateUtils, ValidationUtils, logger, PythonIntegration } from '../utils';

export class IndexController {
    private pythonIntegration: PythonIntegration;
    
    constructor() {
        this.pythonIntegration = new PythonIntegration();
    }

    public home(req: Request, res: Response): void {
        logger('Acceso a la página principal');
        res.send({
            message: "Bienvenido a la API de Mi Proyecto",
            currentDate: DateUtils.getCurrentDate(),
            version: "1.0.0"
        });
    }

    public about(req: Request, res: Response): void {
        logger('Acceso a la página acerca de');
        res.send({
            message: "Acerca de Mi Proyecto - Aplicación web y ciencia de datos",
            tech: {
                backend: "TypeScript + Express.js",
                dataScienceML: "Python + scikit-learn, pandas, etc."
            },
            contact: "ejemplo@correo.com"
        });
    }
    
    public async runDataAnalysis(req: Request, res: Response): Promise<void> {
        try {
            const { datasetName } = req.params;
            
            if (!ValidationUtils.isNotEmpty(datasetName)) {
                res.status(400).json({ error: 'Nombre de dataset no válido' });
                return;
            }
            
            logger(`Ejecutando análisis de datos para: ${datasetName}`);
            const resultado = await this.pythonIntegration.executeScript('1_fundamentos/introduccion.py', [datasetName]);
            
            res.json({
                success: true,
                resultado,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            console.error('Error al ejecutar análisis de datos:', error);
            res.status(500).json({ error: 'Error al procesar la solicitud' });
        }
    }
    
    public async runMachineLearning(req: Request, res: Response): Promise<void> {
        try {
            const { datasetName } = req.params;
            
            if (!ValidationUtils.isNotEmpty(datasetName)) {
                res.status(400).json({ error: 'Nombre de dataset no válido' });
                return;
            }
            
            logger(`Ejecutando modelo de clasificación para: ${datasetName}`);
            
            // Ejecutar el script de clasificación con el nombre del dataset
            const resultadoStr = await this.pythonIntegration.executeScript(
                '3_machine_learning/supervisado/clasificacion_basica.py', 
                [datasetName]
            );
            
            // Convertir la salida string a JSON
            let resultado;
            try {
                resultado = JSON.parse(resultadoStr);
            } catch (e) {
                logger(`Error al parsear resultado JSON: ${resultadoStr}`);
                resultado = { output: resultadoStr };
            }
            
            res.json({
                success: true,
                modelo: "Random Forest",
                resultado,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            console.error('Error al ejecutar modelo de machine learning:', error);
            res.status(500).json({ 
                error: 'Error al procesar la solicitud de machine learning',
                message: error instanceof Error ? error.message : 'Error desconocido'
            });
        }
    }
}

export default IndexController;