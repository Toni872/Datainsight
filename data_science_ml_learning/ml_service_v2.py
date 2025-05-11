#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Servicio API mejorado para ML usando FastAPI
Este script crea un API REST avanzado para servir modelos de machine learning
"""

import os
import sys
import json
import time
import base64
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from joblib import dump, load
import matplotlib.pyplot as plt

# Intentar importar dependencias opcionales
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost no está disponible")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM no está disponible")

# Cargar módulos propios
# Asegurar que podemos importar desde directorios relevantes
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '3_machine_learning', 'supervisado'))

# Verify if the file exists and provide a fallback
if not os.path.exists(os.path.join(script_dir, '3_machine_learning', 'supervisado', 'model_trainer.py')):
    print("Warning: 'model_trainer.py' not found in '3_machine_learning/supervisado'. Ensure the file exists.")
sys.path.append(os.path.join(script_dir, '3_machine_learning', 'no_supervisado'))
sys.path.append(os.path.join(script_dir, '5_especializacion', 'series_temporales'))

# Agregar patrón de búsqueda de módulos de utilidades
sys.path.append(os.path.join(script_dir, 'utils'))

# Importar sistema de caché y visualización
try:
    from utils.cache_manager import CacheManager
    from utils.visualization import AdvancedVisualizer
    CACHE_AVAILABLE = True
    print("Sistema de caché cargado correctamente")
    
    # Inicializar el sistema de caché y el visualizador
    cache_manager = CacheManager()
    visualizer = AdvancedVisualizer(use_cache=True)
except ImportError as e:
    CACHE_AVAILABLE = False
    print(f"No se pudo cargar el sistema de caché: {e}")
    cache_manager = None
    visualizer = None

# Importar módulos propios
try:
    try:
        from dataset_loader import DatasetProcessor
    except ImportError:
        print("Warning: 'dataset_loader' module not found. Ensure it exists in the correct directory.")
        DatasetProcessor = None
    from model_trainer import ModelTrainer
    from advanced_models import AdvancedModelTrainer
    from model_evaluator import ModelEvaluator
    # Import para clustering avanzado
    from advanced_clustering import UnsupervisedModelTrainer
    # Import para series temporales
    from time_series_analyzer import TimeSeriesAnalyzer
except ImportError as e:
    print(f"Error importando módulos propios: {e}")
    # Crear rutas si no existen
    os.makedirs(os.path.join(script_dir, '3_machine_learning', 'supervisado'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, '3_machine_learning', 'no_supervisado'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, '5_especializacion', 'series_temporales'), exist_ok=True)

# Crear la aplicación FastAPI
app = FastAPI(
    title="DataInsight AI - Servicio ML Avanzado",
    description="API mejorada para servir modelos avanzados de machine learning",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, limitar a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos para la API
class PredictionRequest(BaseModel):
    features: List[float]
    model_params: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: Any
    prediction_proba: Optional[List[float]] = None
    model_name: str
    confidence: Optional[float] = None
    execution_time_ms: float
    additional_info: Optional[Dict[str, Any]] = None

class TrainRequest(BaseModel):
    dataset_id: str
    model_type: str
    model_name: str
    test_size: float = 0.2
    model_params: Optional[Dict[str, Any]] = None
    optimize_hyperparams: bool = False

class DatasetUploadRequest(BaseModel):
    dataset_name: str
    target_column: Optional[str] = None
    categorical_columns: Optional[List[str]] = None

class ClusteringRequest(BaseModel):
    dataset_id: str
    algorithm: str
    n_clusters: Optional[int] = None
    model_params: Optional[Dict[str, Any]] = None
    dimensionality_reduction: Optional[str] = None

class EvaluationRequest(BaseModel):
    model_id: str
    evaluation_type: str
    params: Optional[Dict[str, Any]] = None

class CorrelationRequest(BaseModel):
    """Estructura de datos para solicitar una matriz de correlación"""
    data: Dict[str, List[float]] = Field(..., description="Datos en formato {variable: [valores]}")
    method: str = Field("pearson", description="Método de correlación: pearson, spearman o kendall")
    title: str = Field("Matriz de Correlación", description="Título del gráfico")

class FeatureImportanceRequest(BaseModel):
    """Estructura de datos para solicitar un gráfico de importancia de características"""
    importance_values: List[float] = Field(..., description="Valores de importancia de cada característica")
    feature_names: List[str] = Field(None, description="Nombres de las características")
    top_n: int = Field(None, description="Mostrar solo las top N características")
    title: str = Field("Importancia de Características", description="Título del gráfico")
    horizontal: bool = Field(True, description="Si True, barras horizontales")

# Crear directorios de almacenamiento si no existen
MODELS_DIR = os.path.join(script_dir, 'models')
DATASETS_DIR = os.path.join(script_dir, 'datasets')
EVALUATIONS_DIR = os.path.join(script_dir, 'evaluations')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)

# Almacenamiento en memoria para modelos y datasets
trained_models = {}
datasets_cache = {}

# Inicializar procesadores y evaluadores
dataset_processor = None
model_trainer = None
advanced_trainer = None
model_evaluator = None

# Función para inicializar componentes
def initialize_components():
    global dataset_processor, model_trainer, advanced_trainer, model_evaluator
    
    try:
        dataset_processor = DatasetProcessor()
        model_trainer = ModelTrainer()
        advanced_trainer = AdvancedModelTrainer()
        model_evaluator = ModelEvaluator()
        print("Componentes inicializados correctamente")
        return True
    except Exception as e:
        print(f"Error inicializando componentes: {e}")
        return False

# Función para cargar modelo predefinido de Iris
def train_iris_model():
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        # Cargar el dataset iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar un clasificador Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        accuracy = model.score(X_test, y_test)
        
        # Guardar en el diccionario de modelos
        trained_models["iris_classifier"] = {
            "model": model,
            "feature_names": iris.feature_names,
            "target_names": iris.target_names,
            "accuracy": accuracy,
            "model_type": "classification"
        }
        
        print(f"Modelo iris_classifier entrenado con precisión: {accuracy:.4f}")
    except Exception as e:
        print(f"Error entrenando modelo iris: {e}")

# Cargar datos y modelos al inicio
@app.on_event("startup")
async def startup_event():
    print("Iniciando servicio ML...")
    
    # Inicializar componentes
    if not initialize_components():
        print("Usando configuración mínima debido a errores en la inicialización de componentes")
    
    # Entrenar modelo de ejemplo
    train_iris_model()
    
    print("Servicio ML listo")

# Rutas de la API

@app.get("/")
async def root():
    """Endpoint raíz para verificar el estado del servicio"""
    components_status = {
        "dataset_processor": dataset_processor is not None,
        "model_trainer": model_trainer is not None,
        "advanced_trainer": advanced_trainer is not None,
        "model_evaluator": model_evaluator is not None
    }
    
    return {
        "status": "online", 
        "service": "DataInsight AI ML Service Avanzado", 
        "version": "2.0.0",
        "components_status": components_status,
        "models_loaded": len(trained_models)
    }

@app.get("/models")
async def get_available_models():
    """Retorna la lista de modelos disponibles"""
    # Obtener modelos disponibles para entrenamiento
    available_training_models = {}
    
    if model_trainer is not None:
        available_training_models["classification"] = list(model_trainer.classification_models.keys())
        available_training_models["regression"] = list(model_trainer.regression_models.keys())
    
    if advanced_trainer is not None:
        advanced_classification = list(advanced_trainer.classification_models.keys())
        advanced_regression = list(advanced_trainer.regression_models.keys())
        
        if "classification" in available_training_models:
            available_training_models["classification"].extend(advanced_classification)
        else:
            available_training_models["classification"] = advanced_classification
            
        if "regression" in available_training_models:
            available_training_models["regression"].extend(advanced_regression)
        else:
            available_training_models["regression"] = advanced_regression
    
    return {
        "trained_models": list(trained_models.keys()),
        "models_info": {
            name: {
                "feature_names": info.get("feature_names"),
                "target_names": info.get("target_names"),
                "accuracy": info.get("accuracy"),
                "model_type": info.get("model_type", "unknown")
            } for name, info in trained_models.items()
        },
        "available_training_models": available_training_models
    }

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    """Realiza una predicción usando un modelo entrenado"""
    # Verificar si el modelo existe
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")
    
    # Obtener modelo y metadatos
    model_data = trained_models[model_name]
    model = model_data["model"]
    
    # Verificar número de características
    features = np.array(request.features).reshape(1, -1)
    expected_features = len(model_data.get("feature_names", []))
    
    if expected_features > 0 and features.shape[1] != expected_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Número incorrecto de características. Se esperaban {expected_features}, se recibieron {features.shape[1]}"
        )
    
    # Realizar predicción
    start_time = time.time()
    
    try:
        prediction = model.predict(features)[0]
        
        # Convertir a tipo nativo Python para serialización JSON
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        
        response = {
            "prediction": prediction,
            "model_name": model_name,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "additional_info": {}
        }
        
        # Añadir probabilidades si están disponibles
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features)[0].tolist()
            response["prediction_proba"] = prediction_proba
            
            # Para clasificación, obtener etiquetas de las clases
            if "target_names" in model_data:
                class_probabilities = {
                    model_data["target_names"][i]: float(prob)
                    for i, prob in enumerate(prediction_proba)
                }
                response["additional_info"]["class_probabilities"] = class_probabilities
            
            # Calcular confianza
            response["confidence"] = float(max(prediction_proba))
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/datasets")
async def list_datasets():
    """Lista todos los datasets disponibles"""
    if dataset_processor is None:
        raise HTTPException(status_code=500, detail="Procesador de datasets no disponible")
    
    try:
        # Obtener datasets estándar
        standard_datasets = list(dataset_processor.standard_datasets.keys())
        
        # Buscar datasets personalizados
        custom_datasets = []
        for file in os.listdir(DATASETS_DIR):
            if file.endswith('.csv'):
                custom_datasets.append(file[:-4])  # Eliminar extensión .csv
        
        return {
            "standard_datasets": standard_datasets,
            "custom_datasets": custom_datasets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar datasets: {str(e)}")

@app.get("/datasets/{dataset_id}/info")
async def get_dataset_info(dataset_id: str):
    """Obtiene información de un dataset específico"""
    if dataset_processor is None:
        raise HTTPException(status_code=500, detail="Procesador de datasets no disponible")
    
    try:
        info = dataset_processor.get_dataset_info(dataset_id)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")

@app.post("/models/train")
async def train_model(request: TrainRequest):
    """Entrena un nuevo modelo con los parámetros especificados"""
    if model_trainer is None or dataset_processor is None:
        raise HTTPException(status_code=500, detail="Componentes de entrenamiento no disponibles")
    
    try:
        # Preparar datos
        data = dataset_processor.prepare_data(
            request.dataset_id, 
            test_size=request.test_size
        )
        
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        
        # Determinar qué tipo de trainer usar
        use_advanced = request.model_name in (
            list(advanced_trainer.classification_models.keys()) + 
            list(advanced_trainer.regression_models.keys())
        )
        
        # Crear ID para el modelo
        model_id = f"{request.dataset_id}_{request.model_name}_{int(time.time())}"
        
        # Entrenar modelo según tipo
        if request.model_type == "classification":
            if use_advanced and advanced_trainer is not None:
                # Usar entrenamiento avanzado
                results = advanced_trainer.train_model_with_evaluation(
                    request.model_name, 
                    "classification",
                    data["X_train"], 
                    data["y_train"], 
                    data["X_test"], 
                    data["y_test"],
                    feature_names=data["feature_names"],
                    class_names=data.get("class_names"),
                    optimize=request.optimize_hyperparams
                )
                
                # Guardar modelo en memoria
                trained_models[model_id] = {
                    "model": advanced_trainer.models_dict[request.model_name],
                    "feature_names": data["feature_names"],
                    "target_names": data.get("class_names"),
                    "accuracy": results["accuracy"],
                    "model_type": "classification"
                }
                
                # Guardar modelo en disco
                model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
                dump(results, model_path)
                
            else:
                # Usar entrenamiento básico
                models_to_train = [request.model_name]
                results = model_trainer.train_models(
                    data_file,
                    "classification", 
                    models_to_train
                )
                
                if "error" in results:
                    raise HTTPException(status_code=500, detail=results["error"])
                
                # Guardar modelo en memoria
                model_results = results["models"][request.model_name]
                trained_models[model_id] = {
                    "model": model_results["model"],
                    "feature_names": data["feature_names"],
                    "target_names": data.get("class_names"),
                    "accuracy": model_results["accuracy"],
                    "model_type": "classification"
                }
                
                # Devolver resultados
                results = model_results
                
        elif request.model_type == "regression":
            # Proceso similar para regresión...
            pass
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de modelo no válido: {request.model_type}")
        
        # Devolver resultados
        return {
            "model_id": model_id,
            **results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@app.post("/clustering")
async def perform_clustering(request: ClusteringRequest):
    """Realiza clustering en un dataset"""
    if dataset_processor is None:
        raise HTTPException(status_code=500, detail="Procesador de datasets no disponible")
    
    try:
        # Cargar dataset
        from sklearn.datasets import load_iris  # Temporal
        X = load_iris().data  # Temporal
        
        # Realizar clustering
        results = {"success": True, "clusters": [0, 1, 2]}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en clustering: {str(e)}")

@app.post("/models/{model_id}/evaluate")
async def evaluate_model(model_id: str, request: EvaluationRequest):
    """Evalúa un modelo con diferentes métricas"""
    if model_evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluador de modelos no disponible")
    
    try:
        # Implementar lógica de evaluación
        results = {"success": True}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {str(e)}")

@app.get("/status")
async def get_status():
    """
    Obtiene el estado del servicio ML y verifica disponibilidad de modelos
    """
    try:
        import sklearn
        scipy_version = sys.modules.get('scipy').__version__ if 'scipy' in sys.modules else "No instalado"
        pandas_version = pd.__version__
        numpy_version = np.__version__
        
        # Verificar modelos disponibles
        available_models = {
            'classification': [],
            'regression': [],
            'clustering': [],
            'timeseries': []
        }
        
        # Modelos supervisados
        try:
            advanced_model_trainer = AdvancedModelTrainer()
            available_models['classification'] = list(advanced_model_trainer.classification_models.keys())
            available_models['regression'] = list(advanced_model_trainer.regression_models.keys())
        except:
            pass
            
        # Modelos no supervisados
        try:
            from advanced_clustering import UnsupervisedModelTrainer
            unsupervised_trainer = UnsupervisedModelTrainer()
            available_models['clustering'] = list(unsupervised_trainer.clustering_models.keys())
        except:
            pass
            
        # Modelos de series temporales
        try:
            from time_series_analyzer import TimeSeriesAnalyzer
            ts_analyzer = TimeSeriesAnalyzer()
            available_models['timeseries'] = ts_analyzer.get_available_methods()['forecasting']
        except:
            pass
        
        return {
            "status": "online",
            "version": "v2.0.0",
            "dependencies": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "numpy": numpy_version,
                "pandas": pandas_version,
                "scikit-learn": sklearn.__version__,
                "scipy": scipy_version,
                "xgboost": "Instalado" if XGBOOST_AVAILABLE else "No instalado",
                "lightgbm": "Instalado" if LIGHTGBM_AVAILABLE else "No instalado"
            },
            "available_models": available_models
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/models/available")
async def get_available_models():
    """
    Obtiene los modelos disponibles y sus capacidades
    """
    try:
        # Modelos de clasificación y regresión
        model_trainer = AdvancedModelTrainer()
        classification_models = list(model_trainer.classification_models.keys())
        regression_models = list(model_trainer.regression_models.keys())
        
        # Modelos de clustering
        try:
            from advanced_clustering import UnsupervisedModelTrainer
            unsupervised_trainer = UnsupervisedModelTrainer()
            clustering_models = list(unsupervised_trainer.clustering_models.keys())
            dimension_reduction = (
                list(unsupervised_trainer.decomposition_models.keys()) + 
                list(unsupervised_trainer.manifold_models.keys())
            )
        except Exception as e:
            print(f"Error cargando modelos de clustering: {e}")
            clustering_models = []
            dimension_reduction = []
        
        # Modelos de series temporales
        try:
            from time_series_analyzer import TimeSeriesAnalyzer
            ts_analyzer = TimeSeriesAnalyzer()
            ts_methods = ts_analyzer.get_available_methods()
        except Exception as e:
            print(f"Error cargando modelos de series temporales: {e}")
            ts_methods = {
                'analysis': ['decomposition', 'stationarity'],
                'forecasting': ['arima', 'sarima']
            }
        
        return {
            "models": {
                "classification": classification_models,
                "regression": regression_models,
                "clustering": {
                    "algorithms": clustering_models,
                    "dimension_reduction": dimension_reduction
                },
                "timeseries": {
                    "analysis": ts_methods['analysis'],
                    "forecasting": ts_methods['forecasting']
                }
            },
            "capabilities": {
                "hyperparameter_optimization": True,
                "feature_importance": True,
                "cross_validation": True,
                "ensemble_methods": True
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/clustering/perform")
async def perform_clustering(
    dataset_id: str = Body(..., embed=True),
    algorithm: str = Body(..., embed=True),
    n_clusters: Optional[int] = Body(3, embed=True),
    dimensionality_reduction: Optional[str] = Body(None, embed=True),
    model_params: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Realiza clustering en un dataset
    """
    try:
        # Importar el módulo de clustering
        try:
            from advanced_clustering import UnsupervisedModelTrainer
            unsupervised_trainer = UnsupervisedModelTrainer()
        except ImportError as e:
            return {"success": False, "error": f"Módulo de clustering no disponible: {str(e)}"}
        
        # Cargar dataset
        dataset_processor = DatasetProcessor()
        X, _ = dataset_processor.load_dataset_by_id(dataset_id)
        
        if X is None:
            return {"success": False, "error": f"Dataset '{dataset_id}' no encontrado"}
        
        # Configurar parámetros
        params = model_params or {}
        if n_clusters and algorithm in ['kmeans', 'agglomerative', 'gaussian_mixture']:
            params['n_clusters'] = n_clusters
        
        # Aplicar reducción de dimensionalidad si se solicita
        if dimensionality_reduction:
            dimension_result = unsupervised_trainer.reduce_dimensions(
                X, method=dimensionality_reduction, n_components=2
            )
            X_transformed = dimension_result['transformed_data']
            
            # Realizar clustering
            result = unsupervised_trainer.perform_clustering(
                X_transformed, algorithm, **params, visualize=True
            )
            
            # Añadir información de la reducción
            result['dimensionality_reduction'] = {
                'method': dimensionality_reduction,
                'explained_variance': dimension_result.get('explained_variance_ratio')
            }
        else:
            # Realizar clustering directamente
            result = unsupervised_trainer.perform_clustering(
                X, algorithm, **params, visualize=True
            )
        
        # Eliminar objetos no serializables
        if 'model' in result:
            del result['model']
        if 'scaler' in result:
            del result['scaler']
        
        # Convertir arrays numpy a listas
        for key, value in result.items():
            if hasattr(value, 'tolist'):
                result[key] = value.tolist()
        
        return {
            "success": True,
            **result
        }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/timeseries/analyze")
async def analyze_time_series(
    dataset_id: str = Body(..., embed=True),
    analysis_type: str = Body(..., embed=True),
    params: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Realiza análisis de series temporales
    """
    try:
        # Importar el módulo de series temporales
        try:
            from time_series_analyzer import TimeSeriesAnalyzer
            ts_analyzer = TimeSeriesAnalyzer()
        except ImportError as e:
            return {"success": False, "error": f"Módulo de series temporales no disponible: {str(e)}"}
        
        # Cargar dataset
        dataset_processor = DatasetProcessor()
        data = dataset_processor.load_dataset_by_id_as_df(dataset_id)
        
        if data is None:
            return {"success": False, "error": f"Dataset '{dataset_id}' no encontrado"}
        
        # Extraer parámetros
        params = params or {}
        date_col = params.get('date_column')
        value_col = params.get('value_column')
        forecast_periods = params.get('forecast_periods', 10)
        
        # Si no se proporcionan columnas, intentar inferirlas
        if not date_col or not value_col:
            # Intentar encontrar columna de fecha
            for col in data.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'día' in col.lower() or 'fecha' in col.lower():
                    date_col = col
                    break
            
            # Intentar encontrar columna numérica para valores
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
        
        if not date_col or not value_col:
            return {
                "success": False, 
                "error": "No se pudieron determinar las columnas de fecha y valor"
            }
        
        # Preparar serie temporal
        try:
            # Convertir a datetime
            data[date_col] = pd.to_datetime(data[date_col])
            # Ordenar por fecha
            data = data.sort_values(by=date_col)
            # Crear serie temporal
            ts_data = pd.Series(data[value_col].values, index=data[date_col])
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error al crear la serie temporal: {str(e)}"
            }
        
        # Realizar el análisis según el tipo
        if analysis_type == 'decomposition':
            # Descomponer serie temporal
            result = ts_analyzer.decompose_time_series(ts_data)
            
            return {
                "success": True,
                "components": {
                    "trend": result['trend'].to_dict(),
                    "seasonal": result['seasonal'].to_dict(),
                    "residual": result['residual'].to_dict()
                },
                "period": result['period'],
                "model": result['model'],
                "visualization": result['visualization']
            }
            
        elif analysis_type == 'stationarity':
            # Analizar estacionariedad
            result = ts_analyzer.check_stationarity(ts_data)
            
            return {
                "success": True,
                "is_stationary": result['is_stationary'],
                "p_value": result['p_value'],
                "test_statistic": result['test_statistic'],
                "critical_values": result['critical_values']
            }
            
        elif analysis_type.startswith('forecast_'):
            # Extraer método de pronóstico
            forecast_method = analysis_type.split('_')[1]
            
            if forecast_method == 'arima':
                # Pronóstico ARIMA
                order = params.get('order')
                seasonal_order = params.get('seasonal_order')
                
                result = ts_analyzer.forecast_arima(
                    ts_data, 
                    periods=forecast_periods,
                    order=order,
                    seasonal_order=seasonal_order
                )
                
                return {
                    "success": True,
                    "forecast_data": result['forecast_data'],
                    "metrics": result['metrics'],
                    "model_info": {
                        "type": result['model_type'],
                        "order": result['order'],
                        "seasonal_order": result['seasonal_order'],
                        "aic": result['aic'],
                        "bic": result['bic']
                    },
                    "visualization": result['visualization']
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Método de pronóstico '{forecast_method}' no implementado"
                }
        else:
            return {
                "success": False,
                "error": f"Tipo de análisis '{analysis_type}' no reconocido"
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/models/evaluate")
async def evaluate_model(
    model_id: str = Body(..., embed=True),
    visualization_type: str = Body(..., embed=True),
    params: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Evalúa un modelo y genera visualizaciones
    """
    try:
        # Importar evaluador
        model_evaluator = ModelEvaluator()
        
        # Cargar modelo
        model_trainer = ModelTrainer()
        model_info = model_trainer.load_model(model_id)
        
        if model_info is None:
            return {"success": False, "error": f"Modelo '{model_id}' no encontrado"}
        
        # Obtener datos del dataset original
        dataset_processor = DatasetProcessor()
        X, y = dataset_processor.load_dataset_by_id(model_info['dataset_id'])
        
        if X is None or y is None:
            return {"success": False, "error": f"Dataset del modelo no encontrado"}
        
        # Dividir en entrenamiento y prueba
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = model_info['model']
        
        # Generar visualización según el tipo
        if visualization_type == 'learning_curve':
            results, img_buffer = model_evaluator.generate_learning_curves(
                model, X, y, cv=params.get('cv', 5) if params else 5
            )
            
            # Codificar imagen
            import base64
            img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            return {
                "success": True,
                "visualization": img_str,
                "metrics": {
                    "train_sizes": results['train_sizes'],
                    "train_mean": results['train_scores_mean'],
                    "test_mean": results['test_scores_mean']
                },
                "explanation": (
                    "La curva de aprendizaje muestra cómo el rendimiento del modelo mejora "
                    "con más datos de entrenamiento. Una brecha grande entre el rendimiento "
                    "de entrenamiento y validación indica sobreajuste."
                )
            }
            
        elif visualization_type == 'feature_importance':
            # Obtener nombres de características
            feature_names = params.get('feature_names') if params else None
            if not feature_names and hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            elif not feature_names:
                feature_names = [f'Característica {i+1}' for i in range(X.shape[1])]
            
            importances, img_buffer = model_evaluator.evaluate_feature_importance(
                model, X, y, feature_names=feature_names
            )
            
            # Codificar imagen
            import base64
            img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            return {
                "success": True,
                "visualization": img_str,
                "feature_importances": importances,
                "explanation": (
                    "La importancia de características muestra qué variables tienen mayor "
                    "impacto en las predicciones del modelo. Las características con "
                    "mayor importancia son más influyentes."
                )
            }
            
        elif visualization_type == 'roc_curve' and model_info['task_type'] == 'classification':
            # Obtener probabilidades
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                
                # Obtener nombres de clases si están disponibles
                class_names = params.get('class_names') if params else None
                
                results, img_buffer = model_evaluator.plot_roc_curves(
                    y_test, y_pred_proba, class_names=class_names
                )
                
                # Codificar imagen
                import base64
                img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
                
                return {
                    "success": True,
                    "visualization": img_str,
                    "metrics": results,
                    "explanation": (
                        "La curva ROC muestra el rendimiento de clasificación en diferentes "
                        "umbrales. Un área bajo la curva (AUC) más cercana a 1 indica mejor "
                        "capacidad de discriminación del modelo."
                    )
                }
            else:
                return {
                    "success": False,
                    "error": "El modelo no admite probabilidades para generar curvas ROC"
                }
        
        elif visualization_type == 'confusion_matrix' and model_info['task_type'] == 'classification':
            # Generar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular matriz de confusión
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(y_test, y_pred)
            
            # Crear visualización
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues')
            plt.title('Matriz de Confusión')
            
            # Guardar figura en buffer
            from io import BytesIO
            import base64
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "success": True,
                "visualization": img_str,
                "confusion_matrix": cm.tolist(),
                "explanation": (
                    "La matriz de confusión muestra las predicciones correctas e incorrectas "
                    "por clase. La diagonal representa las predicciones acertadas, mientras "
                    "que el resto son errores."
                )
            }
            
        elif visualization_type == 'residuals' and model_info['task_type'] == 'regression':
            # Generar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular residuos
            residuals = y_test - y_pred
            
            # Crear visualización
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_pred, residuals)
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_xlabel('Predicciones')
            ax.set_ylabel('Residuos')
            ax.set_title('Gráfico de Residuos')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Guardar figura en buffer
            from io import BytesIO
            import base64
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            # Calcular métricas de residuos
            import numpy as np
            metrics = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'max_residual': float(np.max(residuals)),
                'min_residual': float(np.min(residuals))
            }
            
            return {
                "success": True,
                "visualization": img_str,
                "metrics": metrics,
                "explanation": (
                    "El gráfico de residuos muestra la diferencia entre valores reales y predicciones. "
                    "Un buen modelo tiene residuos distribuidos aleatoriamente alrededor de cero, sin "
                    "patrones visibles."
                )
            }
            
        else:
            return {
                "success": False,
                "error": f"Tipo de visualización '{visualization_type}' no compatible con el modelo"
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/cache/stats", tags=["Caché"])
async def get_cache_stats():
    """
    Obtiene estadísticas del sistema de caché
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sistema de caché no disponible")
    
    try:
        stats = cache_manager.get_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas de caché: {str(e)}")


@app.post("/cache/clear", tags=["Caché"])
async def clear_cache():
    """
    Limpia el sistema de caché
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sistema de caché no disponible")
    
    try:
        cache_manager.clear()
        return {
            "status": "success",
            "message": "Caché limpiada correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar caché: {str(e)}")

@app.post("/visualization/correlation_matrix", tags=["Visualización"])
async def create_correlation_matrix(request: CorrelationRequest):
    """
    Crea una matriz de correlación a partir de los datos proporcionados
    Utiliza el sistema de caché para optimizar el rendimiento
    """
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame(request.data)
        
        if CACHE_AVAILABLE and visualizer:
            # Usar el sistema de caché y visualización avanzada
            viz_buffer = visualizer.correlation_matrix(
                data=df,
                method=request.method,
                title=request.title
            )
        else:
            # Fallback a implementación básica sin caché
            corr_matrix = df.corr(method=request.method)
            
            plt.figure(figsize=(10, 8))
            plt.matshow(corr_matrix, fignum=1)
            plt.colorbar()
            plt.title(request.title)
            
            # Añadir etiquetas
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            
            # Añadir valores
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center",
                           color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
            
            plt.tight_layout()
            
            # Guardar en buffer
            viz_buffer = BytesIO()
            plt.savefig(viz_buffer, format='png')
            plt.close()
            viz_buffer.seek(0)
        
        # Devolver imagen como respuesta
        return StreamingResponse(viz_buffer, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear matriz de correlación: {str(e)}")

@app.post("/visualization/feature_importance", tags=["Visualización"])
async def create_feature_importance(request: FeatureImportanceRequest):
    """
    Crea un gráfico de importancia de características
    Utiliza el sistema de caché para optimizar el rendimiento
    """
    try:
        if len(request.importance_values) == 0:
            raise HTTPException(status_code=400, detail="La lista de valores de importancia no puede estar vacía")
        
        # Si no se proporcionan nombres, usar valores genéricos
        if request.feature_names is None or len(request.feature_names) != len(request.importance_values):
            feature_names = [f"Feature {i+1}" for i in range(len(request.importance_values))]
        else:
            feature_names = request.feature_names
        
        if CACHE_AVAILABLE and visualizer:
            # Usar el sistema de caché y visualización avanzada
            viz_buffer = visualizer.feature_importance(
                importance_values=np.array(request.importance_values),
                feature_names=feature_names,
                top_n=request.top_n,
                title=request.title,
                horizontal=request.horizontal
            )
        else:
            # Fallback a implementación básica sin caché
            # Crear DataFrame para facilitar el ordenamiento
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': request.importance_values
            }).sort_values('importance', ascending=False)
            
            # Filtrar por top_n si es necesario
            if request.top_n is not None and request.top_n < len(importance_df):
                importance_df = importance_df.head(request.top_n)
            
            # Crear figura
            plt.figure(figsize=(10, 6))
            
            # Crear gráfico de barras
            if request.horizontal:
                importance_df = importance_df.sort_values('importance')
                plt.barh(importance_df['feature'], importance_df['importance'])
                plt.xlabel('Importancia')
                plt.ylabel('Característica')
            else:
                plt.bar(importance_df['feature'], importance_df['importance'])
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Característica')
                plt.ylabel('Importancia')
            
            plt.title(request.title)
            plt.tight_layout()
            
            # Guardar en buffer
            viz_buffer = BytesIO()
            plt.savefig(viz_buffer, format='png')
            plt.close()
            viz_buffer.seek(0)
        
        # Devolver imagen como respuesta
        return StreamingResponse(viz_buffer, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear gráfico de importancia: {str(e)}")

# Función principal para ejecutar el servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ml_service:app", host="0.0.0.0", port=port, reload=True)
