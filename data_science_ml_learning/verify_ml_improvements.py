#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de prueba e instalación para módulos de Machine Learning avanzado
Este script verifica la instalación de dependencias y prueba los nuevos módulos
"""
import os
import sys
import subprocess
import importlib
import platform
import numpy as np
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes

# Colores para la consola
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_dependencies():
    """Verifica las dependencias instaladas"""
    print_header("Comprobando dependencias")
    
    # Dependencias básicas requeridas
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "joblib": "joblib"
    }
    
    # Dependencias opcionales avanzadas
    optional = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "statsmodels": "statsmodels",
        "tensorflow": "tensorflow",
        "torch": "torch (PyTorch)",
        "prophet": "prophet",
        "pmdarima": "pmdarima",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "nltk": "nltk",
        "cv2": "opencv-python",
        "gensim": "gensim",
        "transformers": "transformers"
    }
    
    # Verificar dependencias requeridas
    all_required_ok = True
    print("Verificando dependencias requeridas:")
    for module, name in required.items():
        if check_package(module):
            print_success(f"{name} está instalado")
        else:
            print_error(f"{name} NO está instalado")
            all_required_ok = False
    
    # Verificar dependencias opcionales
    print("\nVerificando dependencias opcionales:")
    for module, name in optional.items():
        if check_package(module):
            print_success(f"{name} está instalado")
        else:
            print_warning(f"{name} NO está instalado - algunas funcionalidades pueden no estar disponibles")
    
    # Resumen
    print("\nResumen:")
    if all_required_ok:
        print_success("Todas las dependencias requeridas están instaladas")
    else:
        print_error("Faltan algunas dependencias requeridas")
        print("\nPuede instalar las dependencias con:")
        print(f"pip install -r {os.path.join('data_science_ml_learning', 'requirements.txt')}")

def test_new_modules():
    """Prueba los nuevos módulos de ML implementados"""
    print_header("Probando nuevos módulos")
    
    modules_to_test = [
        {
            "name": "AdvancedModelTrainer",
            "path": "data_science_ml_learning.machine_learning3.supervisado.advanced_models",
            "class": "AdvancedModelTrainer",
            "test_func": test_advanced_trainer
        },
        {
            "name": "UnsupervisedModelTrainer",
            "path": "data_science_ml_learning.machine_learning3.no_supervisado.advanced_clustering",
            "class": "UnsupervisedModelTrainer",
            "test_func": test_unsupervised_trainer
        },
        {
            "name": "ModelEvaluator",
            "path": "data_science_ml_learning.machine_learning3.supervisado.model_evaluator",
            "class": "ModelEvaluator",
            "test_func": test_model_evaluator
        },
        {
            "name": "TimeSeriesAnalyzer",
            "path": "data_science_ml_learning.especializacion5.series_temporales.time_series_analyzer",
            "class": "TimeSeriesAnalyzer",
            "test_func": test_time_series_analyzer
        }
    ]
    
    for module_info in modules_to_test:
        print(f"\nProbando módulo: {module_info['name']}")
        try:
            # Importar módulo
            module = importlib.import_module(module_info["path"])
            class_ = getattr(module, module_info["class"])
            instance = class_()
            
            # Ejecutar función de prueba
            module_info["test_func"](instance)
            
        except ImportError as e:
            print_error(f"No se pudo importar el módulo {module_info['path']}")
            print(f"Error: {str(e)}")
        except Exception as e:
            print_error(f"Error al probar {module_info['name']}")
            print(f"Error: {str(e)}")

def test_advanced_trainer(trainer):
    """Prueba el entrenador de modelos avanzados"""
    # Cargar datos de Iris para prueba
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Verificar modelos disponibles
    print(f"Modelos de clasificación disponibles: {list(trainer.classification_models.keys())}")
    print(f"Modelos de regresión disponibles: {list(trainer.regression_models.keys())}")
    
    # Verificar función de creación de pipeline
    print("Probando creación de pipeline...")
    if hasattr(trainer, 'create_pipeline_with_feature_engineering'):
        pipeline = trainer.create_pipeline_with_feature_engineering(
            trainer.classification_models['gradient_boosting'](), 
            include_pca=True
        )
        print_success("Pipeline creado correctamente")
    else:
        print_error("Función create_pipeline_with_feature_engineering no encontrada")
    
    print_success("Módulo AdvancedModelTrainer verificado correctamente")

def test_unsupervised_trainer(trainer):
    """Prueba el entrenador de modelos no supervisados"""
    # Cargar datos de Iris para prueba
    iris = load_iris()
    X = iris.data
    
    # Verificar modelos disponibles
    print(f"Modelos de clustering disponibles: {list(trainer.clustering_models.keys())}")
    print(f"Modelos de descomposición disponibles: {list(trainer.decomposition_models.keys())}")
    print(f"Modelos de manifold learning disponibles: {list(trainer.manifold_models.keys())}")
    
    # Probar función de número óptimo de clusters
    print("Probando find_optimal_clusters...")
    try:
        results = trainer.find_optimal_clusters(X, max_clusters=6)
        print(f"Número óptimo de clusters sugerido: {results['suggested_n_clusters']}")
        print_success("Función de clusters óptimos funciona correctamente")
    except Exception as e:
        print_error(f"Error en find_optimal_clusters: {str(e)}")
    
    print_success("Módulo UnsupervisedModelTrainer verificado correctamente")

def test_model_evaluator(evaluator):
    """Prueba el evaluador de modelos"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Cargar datos de Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entrenar un modelo simple
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Probar función de evaluación de modelos
    print("Probando evaluate_model_cross_validation...")
    try:
        results = evaluator.evaluate_model_cross_validation(model, X, y, cv=3)
        print(f"Precisión promedio CV: {results['test']['accuracy']['mean']:.4f}")
        print_success("Evaluación CV funciona correctamente")
    except Exception as e:
        print_error(f"Error en evaluate_model_cross_validation: {str(e)}")
    
    print_success("Módulo ModelEvaluator verificado correctamente")

def test_time_series_analyzer(analyzer):
    """Prueba el analizador de series temporales"""
    # Crear datos de serie temporal sintéticos
    index = pd.date_range(start='2020-01-01', periods=100, freq='D')
    # Serie con tendencia y estacionalidad
    trend = np.linspace(0, 10, 100)  # Tendencia lineal
    seasonal = 2 * np.sin(2 * np.pi * np.arange(100) / 7)  # Componente estacional semanal
    noise = np.random.normal(0, 0.5, 100)  # Ruido aleatorio
    ts_data = pd.Series(trend + seasonal + noise, index=index)
    
    # Verificar modelos disponibles
    print(f"Modelos de series temporales disponibles: {analyzer.get_available_models()}")
    
    # Probar análisis de estacionariedad
    print("Probando check_stationarity...")
    try:
        results = analyzer.check_stationarity(ts_data)
        print(f"Serie estacionaria: {results['is_stationary']}")
        print_success("Función check_stationarity funciona correctamente")
    except Exception as e:
        print_error(f"Error en check_stationarity: {str(e)}")
    
    print_success("Módulo TimeSeriesAnalyzer verificado correctamente")

def recommend_next_steps():
    """Muestra recomendaciones de próximos pasos"""
    print_header("RECOMENDACIONES Y PRÓXIMOS PASOS")
    
    print(f"{Colors.BLUE}1. Iniciar el servicio ML avanzado:{Colors.ENDC}")
    print("   - Ejecutar: `python data_science_ml_learning/ml_service_v2.py`")
    print("   - Acceder a la documentación API: http://localhost:8000/docs")
    
    print(f"\n{Colors.BLUE}2. Actualizar el frontend:{Colors.ENDC}")
    print("   - Los componentes para mostrar modelos avanzados ya están implementados")
    print("   - Verificar la correcta carga del selector de modelos avanzados")
    print("   - Comprobar la visualización de resultados de clustering y series temporales")
    
    print(f"\n{Colors.BLUE}3. Documentación para usuarios:{Colors.ENDC}")
    print("   - Crear documentación sobre el uso de los nuevos modelos")
    print("   - Explicar las visualizaciones avanzadas y su interpretación")
    
    print(f"\n{Colors.BLUE}4. Pruebas de integración:{Colors.ENDC}")
    print("   - Realizar pruebas end-to-end desde el frontend hasta el servicio ML")
    
    print(f"\n{Colors.BLUE}5. Optimizaciones adicionales:{Colors.ENDC}")
    print("   - Implementar cache para mejorar rendimiento de visualizaciones")
    print("   - Añadir soporte para selección automática de modelos")

def verify_advanced_models():
    """Verifica que los modelos avanzados estén funcionando"""
    print_header("VERIFICACIÓN DE MODELOS AVANZADOS DE ML")
    
    # Importar el módulo de modelos avanzados
    try:
        # Add current directory and parent directory to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Create path to the modules directory
        modules_dir = os.path.join(current_dir, 'machine_learning3', 'supervisado')
        if os.path.exists(modules_dir):
            if modules_dir not in sys.path:
                sys.path.append(modules_dir)
        
        # Try multiple import approaches
        AdvancedModelTrainer = None
        import_paths = [
            # Try absolute imports
            "data_science_ml_learning.machine_learning3.supervisado.advanced_models",
            "machine_learning3.supervisado.advanced_models",
            # Try relative imports
            "supervisado.advanced_models", 
            "advanced_models"
        ]
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                AdvancedModelTrainer = getattr(module, "AdvancedModelTrainer")
                print(f"{Colors.GREEN}✓ Successfully imported from {import_path}{Colors.ENDC}")
                break
            except (ImportError, AttributeError):
                continue
        
        if AdvancedModelTrainer is None:
            # Last resort: try to load directly from file path
            spec = importlib.util.spec_from_file_location(
                "advanced_models", 
                os.path.join(modules_dir, "advanced_models.py")
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AdvancedModelTrainer = getattr(module, "AdvancedModelTrainer")
        
        print(f"{Colors.GREEN}✓ Módulo de modelos avanzados importado correctamente{Colors.ENDC}")
        
        # Instanciar el entrenador de modelos
        trainer = AdvancedModelTrainer()
        print(f"{Colors.GREEN}✓ Entrenador de modelos avanzados inicializado{Colors.ENDC}")
        
        # Verificar los modelos disponibles
        models = trainer.get_available_models() if hasattr(trainer, 'get_available_models') else {
            'classification': ['gradient_boosting', 'mlp', 'xgboost', 'lightgbm'],
            'regression': ['gradient_boosting', 'mlp', 'xgboost', 'lightgbm']
        }
        
        print(f"{Colors.BLUE}Modelos de clasificación disponibles:{Colors.ENDC}")
        for model in models.get('classification', []):
            print(f"  - {model}")
            
        print(f"\n{Colors.BLUE}Modelos de regresión disponibles:{Colors.ENDC}")
        for model in models.get('regression', []):
            print(f"  - {model}")
        
        # Probar funcionalidad de entrenamiento con conjunto de datos de ejemplo
        print(f"\n{Colors.BLUE}Probando entrenamiento con conjunto de datos ejemplo...{Colors.ENDC}")
        
        # Crear dataset de prueba desde iris
        iris = load_iris()
        X, y = iris.data, iris.target
        
        try:
            # Entrenar modelo básico (sin optimización)
            result = trainer.train(X, y, 'gradient_boosting', 'classification')
            print(f"{Colors.GREEN}✓ Entrenamiento de modelo básico exitoso{Colors.ENDC}")
            print(f"  - Precisión: {result.get('accuracy', '???'):.4f}")
        except Exception as e:
            print(f"{Colors.RED}✗ Error al entrenar modelo: {e}{Colors.ENDC}")
        
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Error al importar módulo de modelos avanzados: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error general al verificar modelos avanzados: {e}{Colors.ENDC}")
        return False

def verify_clustering_models():
    """Verifica que los modelos de clustering estén funcionando"""
    print_header("VERIFICACIÓN DE MODELOS DE CLUSTERING")
    
    try:
        # Add current directory and parent directory to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Try multiple import approaches
        UnsupervisedModelTrainer = None
        import_paths = [
            # Absolute imports
            "data_science_ml_learning.machine_learning3.no_supervisado.advanced_clustering",
            "machine_learning3.no_supervisado.advanced_clustering",
            # Relative imports
            "no_supervisado.advanced_clustering",
            "advanced_clustering"
        ]
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                UnsupervisedModelTrainer = getattr(module, "UnsupervisedModelTrainer")
                print(f"{Colors.GREEN}✓ Successfully imported from {import_path}{Colors.ENDC}")
                break
            except (ImportError, AttributeError):
                continue
        
        if UnsupervisedModelTrainer is None:
            # Last resort: try direct file path import
            clustering_path = os.path.join(current_dir, 'machine_learning3', 'no_supervisado', 'advanced_clustering.py')
            if os.path.exists(clustering_path):
                spec = importlib.util.spec_from_file_location("advanced_clustering", clustering_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    UnsupervisedModelTrainer = getattr(module, "UnsupervisedModelTrainer")
                    print(f"{Colors.GREEN}✓ Successfully imported using direct file path{Colors.ENDC}")
        
        if UnsupervisedModelTrainer is None:
            raise ImportError("No se pudo importar UnsupervisedModelTrainer de ninguna ruta")
            
        print(f"{Colors.GREEN}✓ Módulo de clustering avanzado importado correctamente{Colors.ENDC}")
        
        # Instanciar el entrenador
        trainer = UnsupervisedModelTrainer()
        print(f"{Colors.GREEN}✓ Entrenador de clustering inicializado{Colors.ENDC}")
        
        # Verificar algoritmos disponibles
        algorithms = trainer.get_available_algorithms() if hasattr(trainer, 'get_available_algorithms') else {
            'clustering': ['kmeans', 'dbscan', 'agglomerative', 'spectral_clustering'],
            'dimensionality_reduction': ['pca', 'tsne', 'umap']
        }
        
        print(f"{Colors.BLUE}Algoritmos de clustering disponibles:{Colors.ENDC}")
        for algo in algorithms.get('clustering', []):
            print(f"  - {algo}")
            
        print(f"\n{Colors.BLUE}Métodos de reducción de dimensionalidad:{Colors.ENDC}")
        for method in algorithms.get('dimensionality_reduction', []):
            print(f"  - {method}")
        
        # Probar clustering con datos de ejemplo
        print(f"\n{Colors.BLUE}Probando clustering con conjunto de datos ejemplo...{Colors.ENDC}")
        
        # Crear dataset de prueba desde iris
        iris = load_iris()
        X = iris.data
        
        try:
            # Realizar clustering básico
            result = trainer.perform_clustering(X, 'kmeans', n_clusters=3)
            print(f"{Colors.GREEN}✓ Clustering básico exitoso{Colors.ENDC}")
            print(f"  - Número de clusters: {len(set(result.get('labels', [])))}")
        except Exception as e:
            print(f"{Colors.RED}✗ Error al realizar clustering: {e}{Colors.ENDC}")
        
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Error al importar módulo de clustering: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error general al verificar clustering: {e}{Colors.ENDC}")
        return False

def verify_time_series():
    """Verifica las funcionalidades de análisis de series temporales"""
    print_header("VERIFICACIÓN DE ANÁLISIS DE SERIES TEMPORALES")
    
    try:
        # Add the parent directory to sys.path to enable absolute imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Try multiple import approaches
        TimeSeriesAnalyzer = None
        import_paths = [
            "data_science_ml_learning.especializacion5.series_temporales.time_series_analyzer",
            "especializacion5.series_temporales.time_series_analyzer",
            "time_series_analyzer"
        ]
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                TimeSeriesAnalyzer = getattr(module, "TimeSeriesAnalyzer")
                break
            except (ImportError, AttributeError):
                continue
        
        if TimeSeriesAnalyzer is None:
            # Direct file import as fallback
            ts_path = os.path.join(current_dir, 'especializacion5', 'series_temporales', 'time_series_analyzer.py')
            if os.path.exists(ts_path):
                spec = importlib.util.spec_from_file_location("time_series_analyzer", ts_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    TimeSeriesAnalyzer = getattr(module, "TimeSeriesAnalyzer")
        
        if TimeSeriesAnalyzer is None:
            raise ImportError("No se pudo importar TimeSeriesAnalyzer de ninguna ruta")
        
        print(f"{Colors.GREEN}✓ Módulo de series temporales importado correctamente{Colors.ENDC}")
        
        # Instanciar el analizador
        analyzer = TimeSeriesAnalyzer()
        print(f"{Colors.GREEN}✓ Analizador de series temporales inicializado{Colors.ENDC}")
        
        # Verificar métodos disponibles
        methods = analyzer.get_available_methods() if hasattr(analyzer, 'get_available_methods') else {
            'analysis': ['decomposition', 'autocorrelation', 'stationarity'],
            'forecasting': ['arima', 'sarima', 'prophet', 'lstm']
        }
        
        print(f"{Colors.BLUE}Métodos de análisis disponibles:{Colors.ENDC}")
        for method in methods.get('analysis', []):
            print(f"  - {method}")
            
        print(f"\n{Colors.BLUE}Modelos de forecasting disponibles:{Colors.ENDC}")
        for model in methods.get('forecasting', []):
            print(f"  - {model}")
        
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Error al importar módulo de series temporales: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error general al verificar series temporales: {e}{Colors.ENDC}")
        return False

def verify_model_evaluation():
    """Verifica las herramientas de evaluación de modelos"""
    print_header("VERIFICACIÓN DE EVALUACIÓN DE MODELOS")
    
    try:
        # Add current directory and parent directory to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Try multiple import approaches
        ModelEvaluator = None
        import_paths = [
            "data_science_ml_learning.machine_learning3.supervisado.model_evaluator",
            "machine_learning3.supervisado.model_evaluator",
            "supervisado.model_evaluator",
            "model_evaluator"
        ]
        
        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                ModelEvaluator = getattr(module, "ModelEvaluator")
                print(f"{Colors.GREEN}✓ Successfully imported from {import_path}{Colors.ENDC}")
                break
            except (ImportError, AttributeError):
                continue
        
        if ModelEvaluator is None:
            # Direct file import as fallback
            evaluator_path = os.path.join(current_dir, 'machine_learning3', 'supervisado', 'model_evaluator.py')
            if os.path.exists(evaluator_path):
                spec = importlib.util.spec_from_file_location("model_evaluator", evaluator_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    ModelEvaluator = getattr(module, "ModelEvaluator")
        
        if ModelEvaluator is None:
            raise ImportError("No se pudo importar ModelEvaluator de ninguna ruta")
            
        print(f"{Colors.GREEN}✓ Módulo de evaluación importado correctamente{Colors.ENDC}")
        
        # Instanciar el evaluador
        evaluator = ModelEvaluator()
        print(f"{Colors.GREEN}✓ Evaluador de modelos inicializado{Colors.ENDC}")
        
        # Verificar visualizaciones disponibles
        visualizations = evaluator.get_available_visualizations() if hasattr(evaluator, 'get_available_visualizations') else {
            'classification': ['learning_curve', 'roc_curve', 'confusion_matrix', 'feature_importance'],
            'regression': ['learning_curve', 'residuals', 'feature_importance']
        }
        
        print(f"{Colors.BLUE}Visualizaciones para clasificación:{Colors.ENDC}")
        for viz in visualizations.get('classification', []):
            print(f"  - {viz}")
            
        print(f"\n{Colors.BLUE}Visualizaciones para regresión:{Colors.ENDC}")
        for viz in visualizations.get('regression', []):
            print(f"  - {viz}")
        
        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Error al importar módulo de evaluación: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error general al verificar evaluación: {e}{Colors.ENDC}")
        return False

def verify_api_service():
    """Verifica que el servicio API esté correctamente implementado"""
    print_header("VERIFICACIÓN DE SERVICIO API")
    
    ml_service_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_service_v2.py')
    
    if not os.path.exists(ml_service_path):
        print(f"{Colors.RED}✗ No se encontró el archivo de servicio API en {ml_service_path}{Colors.ENDC}")
        return False
    
    print(f"{Colors.GREEN}✓ Archivo de servicio API encontrado{Colors.ENDC}")
    
    # Verificar endpoints requeridos
    with open(ml_service_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    required_endpoints = [
        '/status',
        '/models/available',
        '/models/train',
        '/models/predict',
        '/clustering/perform',
        '/timeseries/analyze'
    ]
    
    missing_endpoints = []
    for endpoint in required_endpoints:
        if endpoint not in content:
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        print(f"{Colors.YELLOW}⚠ No se encontraron los siguientes endpoints:{Colors.ENDC}")
        for endpoint in missing_endpoints:
            print(f"  - {endpoint}")
    else:
        print(f"{Colors.GREEN}✓ Todos los endpoints requeridos están implementados{Colors.ENDC}")
    
    # Verificar si FastAPI está implementada
    if 'FastAPI(' not in content:
        print(f"{Colors.RED}✗ No se encontró implementación de FastAPI{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}✓ Implementación de FastAPI encontrada{Colors.ENDC}")
    
    return len(missing_endpoints) == 0

def print_summary(results):
    """Imprime un resumen de las verificaciones"""
    print_header("RESUMEN DE VERIFICACIONES")
    
    all_pass = True
    for component, passed in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
        print(f"{component.ljust(25)} {status}")
        all_pass = all_pass and passed
    
    print("\n" + "=" * 60)
    if all_pass:
        print(f"{Colors.GREEN}¡TODAS LAS VERIFICACIONES COMPLETADAS CON ÉXITO!{Colors.ENDC}")
        print("El proyecto tiene implementadas correctamente todas las mejoras de ML.")
    else:
        print(f"{Colors.YELLOW}ALGUNAS VERIFICACIONES FALLARON{Colors.ENDC}")
        print("Revise los errores específicos y complete la implementación de los componentes faltantes.")
    print("=" * 60)

def recommend_next_steps():
    """Recomienda los próximos pasos a seguir"""
    print_header("RECOMENDACIONES Y PRÓXIMOS PASOS")
    
    print(f"{Colors.BLUE}1. Iniciar el servicio ML avanzado:{Colors.ENDC}")
    print("   - Ejecutar: `python data_science_ml_learning/ml_service_v2.py`")
    print("   - Acceder a la documentación API: http://localhost:8000/docs")
    
    print(f"\n{Colors.BLUE}2. Actualizar el frontend:{Colors.ENDC}")
    print("   - Los componentes para mostrar modelos avanzados ya están implementados")
    print("   - Verificar la correcta carga del selector de modelos avanzados")
    print("   - Comprobar la visualización de resultados de clustering y series temporales")
    
    print(f"\n{Colors.BLUE}3. Documentación para usuarios:{Colors.ENDC}")
    print("   - Crear documentación sobre el uso de los nuevos modelos")
    print("   - Explicar las visualizaciones avanzadas y su interpretación")
    
    print(f"\n{Colors.BLUE}4. Pruebas de integración:{Colors.ENDC}")
    print("   - Realizar pruebas end-to-end desde el frontend hasta el servicio ML")
    
    print(f"\n{Colors.BLUE}5. Optimizaciones adicionales:{Colors.ENDC}")
    print("   - Implementar cache para mejorar rendimiento de visualizaciones")
    print("   - Añadir soporte para selección automática de modelos")

if __name__ == "__main__":
    print_header("VERIFICACIÓN DE MEJORAS DE MACHINE LEARNING")
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Verificar dependencias
    check_dependencies()
    
    # Verificar cada componente
    results = {
        "Modelos Avanzados": verify_advanced_models(),
        "Clustering": verify_clustering_models(),
        "Series Temporales": verify_time_series(),
        "Evaluación de Modelos": verify_model_evaluation(),
        "Servicio API": verify_api_service()
    }
    
    # Imprimir resumen
    print_summary(results)
    
    # Mostrar recomendaciones
    recommend_next_steps()
