import { Express, Request, Response } from 'express';
import multer from 'multer';
import * as path from 'path';
import IndexController from '../controllers/index';
import { ModelsController } from '../controllers/models.controller';
import { pythonTestController } from '../controllers/python-test.controller';
import { healthCheckMiddleware } from '../middlewares/diagnostics.middleware';

// Crear el directorio de uploads si no existe
const uploadsDir = path.join(__dirname, '../../temp/uploads');
if (!require('fs').existsSync(uploadsDir)) {
    require('fs').mkdirSync(uploadsDir, { recursive: true });
}

// Configurar multer para la carga de archivos
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadsDir);
    },
    filename: function (req, file, cb) {
        // Generar un nombre único para evitar colisiones
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req: Express.Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
        // Solo permitir archivos CSV
        if (file.mimetype !== 'text/csv' && !file.originalname.endsWith('.csv')) {
            return cb(new Error('Solo se permiten archivos CSV'));
        }
        cb(null, true);
    },
    limits: {
        fileSize: 10 * 1024 * 1024 // Limitar a 10MB
    }
});

export const setRoutes = (app: Express): void => {
    const indexController = new IndexController();
    const modelsController = new ModelsController();

    // Ruta de diagnóstico y salud
    app.get('/health', healthCheckMiddleware);
    app.get('/api/health', healthCheckMiddleware);

    // Rutas API básicas
    app.get('/api', indexController.home.bind(indexController));
    app.get('/api/about', indexController.about.bind(indexController));
    
    // Rutas para integración con ciencia de datos (existentes)
    app.get('/api/analyze/:datasetName', indexController.runDataAnalysis.bind(indexController));
    app.get('/api/ml/classify/:datasetName', indexController.runMachineLearning.bind(indexController));
    
    // Rutas para modelos ML (nuevos endpoints)
    app.get('/api/datasets/list', modelsController.listDatasets);
    app.get('/api/datasets/:id', modelsController.getDatasetInfo);
    app.post('/api/datasets/upload', upload.single('file'), modelsController.uploadDataset);
    app.post('/api/models/train', modelsController.trainModels);
    app.post('/api/models/predict', modelsController.predict);
    
    // Nuevas rutas para modelos ML avanzados
    app.get('/api/ml/status', modelsController.getMLServiceStatus);
    app.get('/api/ml/available-models', modelsController.getAvailableModels);
    app.post('/api/ml/train', modelsController.trainAdvancedModel);
    app.post('/api/ml/clustering', modelsController.performClustering);
    app.post('/api/ml/timeseries', modelsController.analyzeTimeSeries);
    app.post('/api/ml/visualize', modelsController.generateModelVisualization);
    
    // Rutas para las vistas HTML
    app.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/index.html'));
    });

    app.get('/dashboard', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/dashboard.html'));
    });

    app.get('/analisis', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/analisis.html'));
    });

    app.get('/modelos', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/modelos.html'));
    });

    app.get('/datasets', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/datasets.html'));
    });

    app.get('/about', (req, res) => {
        res.sendFile(path.join(__dirname, '../../public/about.html'));
    });

    // Rutas API para integración con Python
    app.get('/api/python/test', pythonTestController.testLocalPythonIntegration.bind(pythonTestController));
    app.get('/api/ml/test', pythonTestController.testAzureMLService.bind(pythonTestController));
    app.get('/api/ml/predict/test', pythonTestController.testSimplePrediction.bind(pythonTestController));

    // Ruta 404 para manejar endpoints no encontrados (solo para rutas API)
    app.all('/api/*', (req: Request, res: Response) => {
        res.status(404).json({
            status: 'error',
            message: `Ruta ${req.originalUrl} no encontrada`
        });
    });

    // Ruta 404 para manejar páginas no encontradas
    app.use((req, res) => {
        res.status(404).sendFile(path.join(__dirname, '../../public/index.html'));
    });
};