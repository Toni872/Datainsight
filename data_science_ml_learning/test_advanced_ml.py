#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para probar los modelos avanzados de Machine Learning
"""
import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, fetch_california_housing

# Configurar rutas para importar módulos propios
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '3_machine_learning', 'supervisado'))
sys.path.append(os.path.join(script_dir, '3_machine_learning', 'advanced_models'))  # Added path for advanced_models
sys.path.append(os.path.join(script_dir, '3_machine_learning', 'no_supervisado'))
sys.path.append(os.path.join(script_dir, '5_especializacion', 'series_temporales'))

def test_advanced_models():
    """Prueba los modelos avanzados de clasificación y regresión"""
    print("\n=== PROBANDO MODELOS AVANZADOS ===")
    
    try:
        # Importar el módulo
        from advanced_models import AdvancedModelTrainer
        
        # Crear una instancia
        trainer = AdvancedModelTrainer()
        
        # Cargar datos de ejemplo
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Probar algoritmos
        for algorithm in ['gradient_boosting', 'mlp', 'stacking']:
            print(f"\nProbando algoritmo: {algorithm}")
            try:
                # Entrenar modelo
                result = trainer.train(X, y, algorithm, 'classification')
                
                # Mostrar resultados
                print(f"- Precisión en validación: {result['accuracy']:.4f}")
                print(f"- F1 Score: {result['f1_score']:.4f}")
                
                # Probar predicción
                test_sample = X[0:1]
                prediction = trainer.predict(result['model'], test_sample)
                print(f"- Predicción de muestra: {prediction}")
                
                print(f"✓ {algorithm} funciona correctamente")
            except Exception as e:
                print(f"✗ Error al probar {algorithm}: {e}")
        
        # Probar optimización de hiperparámetros
        print("\nProbando optimización de hiperparámetros...")
        try:
            optimized = trainer.train_with_hyperparameter_tuning(
                X, y, 'gradient_boosting', 'classification'
            )
            print(f"- Mejores parámetros: {optimized['best_params']}")
            print(f"- Precisión optimizada: {optimized['accuracy']:.4f}")
            print("✓ Optimización de hiperparámetros funciona correctamente")
        except Exception as e:
            print(f"✗ Error en optimización: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Error al importar módulos: {e}")
        return False
    except Exception as e:
        print(f"✗ Error general: {e}")
        return False

def test_clustering():
    """Prueba los algoritmos de clustering avanzados"""
    print("\n=== PROBANDO CLUSTERING AVANZADO ===")
    
    try:
        # Importar el módulo
        from advanced_clustering import UnsupervisedModelTrainer
        
        # Crear una instancia
        trainer = UnsupervisedModelTrainer()
        
        # Cargar datos de ejemplo
        iris = load_iris()
        X = iris.data
        
        # Probar algoritmos de clustering
        algorithms = ['kmeans', 'dbscan', 'agglomerative']
        for algorithm in algorithms:
            print(f"\nProbando algoritmo: {algorithm}")
            try:
                # Realizar clustering
                if algorithm == 'kmeans':
                    result = trainer.perform_clustering(X, algorithm, n_clusters=3)
                else:
                    result = trainer.perform_clustering(X, algorithm)
                
                # Mostrar resultados
                print(f"- Número de clusters encontrados: {len(set(result['labels']))}")
                if 'silhouette_score' in result:
                    print(f"- Silhouette score: {result['silhouette_score']:.4f}")
                
                print(f"✓ {algorithm} funciona correctamente")
            except Exception as e:
                print(f"✗ Error al probar {algorithm}: {e}")
        
        # Probar búsqueda del número óptimo de clusters
        print("\nBuscando número óptimo de clusters...")
        try:
            optimal = trainer.find_optimal_clusters(X, max_clusters=10)
            print(f"- Número sugerido de clusters: {optimal['suggested_n_clusters']}")
            print("✓ Búsqueda de clusters óptimos funciona correctamente")
        except Exception as e:
            print(f"✗ Error en búsqueda de clusters óptimos: {e}")
        
        # Probar reducción de dimensionalidad
        print("\nProbando reducción de dimensionalidad...")
        try:
            reduced = trainer.reduce_dimensions(X, method='pca', n_components=2)
            print(f"- Forma original: {X.shape}")
            print(f"- Forma reducida: {reduced['transformed_data'].shape}")
            print("✓ Reducción de dimensionalidad funciona correctamente")
        except Exception as e:
            print(f"✗ Error en reducción de dimensionalidad: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Error al importar módulos: {e}")
        return False
    except Exception as e:
        print(f"✗ Error general: {e}")
        return False

def test_time_series():
    """Prueba las funcionalidades de análisis de series temporales"""
    print("\n=== PROBANDO ANÁLISIS DE SERIES TEMPORALES ===")
    
    try:
        # Importar el módulo
        from time_series_analyzer import TimeSeriesAnalyzer
        
        # Crear una instancia
        analyzer = TimeSeriesAnalyzer()
        
        # Crear datos de serie temporal sintéticos
        index = pd.date_range(start='2020-01-01', periods=100, freq='D')
        # Serie con tendencia y estacionalidad
        trend = np.linspace(0, 10, 100)  # Tendencia lineal
        seasonal = 2 * np.sin(2 * np.pi * np.arange(100) / 7)  # Componente estacional semanal
        noise = np.random.normal(0, 0.5, 100)  # Ruido aleatorio
        ts_data = pd.Series(trend + seasonal + noise, index=index)
        
        # Probar descomposición de serie
        print("\nProbando descomposición de serie temporal...")
        try:
            decomposition = analyzer.decompose_time_series(ts_data)
            print(f"- Componentes extraídos: {list(decomposition.keys())}")
            print("✓ Descomposición funciona correctamente")
        except Exception as e:
            print(f"✗ Error en descomposición: {e}")
        
        # Probar análisis de estacionariedad
        print("\nProbando análisis de estacionariedad...")
        try:
            stationarity = analyzer.check_stationarity(ts_data)
            print(f"- Serie estacionaria: {stationarity['is_stationary']}")
            print(f"- p-value: {stationarity['p_value']:.4f}")
            print("✓ Análisis de estacionariedad funciona correctamente")
        except Exception as e:
            print(f"✗ Error en análisis de estacionariedad: {e}")
        
        # Probar pronóstico ARIMA
        print("\nProbando pronóstico ARIMA...")
        try:
            forecast = analyzer.forecast_arima(ts_data, periods=10)
            print(f"- Pronóstico generado para {len(forecast['forecast'])} periodos")
            print("✓ Pronóstico ARIMA funciona correctamente")
        except Exception as e:
            print(f"✗ Error en pronóstico ARIMA: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Error al importar módulos: {e}")
        return False
    except Exception as e:
        print(f"✗ Error general: {e}")
        return False

def test_model_evaluation():
    """Prueba las herramientas de evaluación de modelos"""
    print("\n=== PROBANDO EVALUACIÓN DE MODELOS ===")
    
    try:
        # Importar módulo
        from model_evaluator import ModelEvaluator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Crear una instancia
        evaluator = ModelEvaluator()
        
        # Cargar datos
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Probar evaluación de modelo
        print("\nProbando evaluación completa...")
        try:
            evaluation = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
            print(f"- Precisión en test: {evaluation['accuracy']:.4f}")
            print(f"- Métricas calculadas: {list(evaluation.keys())}")
            print("✓ Evaluación completa funciona correctamente")
        except Exception as e:
            print(f"✗ Error en evaluación completa: {e}")
        
        # Probar curvas de aprendizaje
        print("\nProbando generación de curvas de aprendizaje...")
        try:
            learning_curve = evaluator.plot_learning_curve(model, X, y, cv=5, return_data=True)
            print(f"- Datos de curva generados correctamente")
            print("✓ Curvas de aprendizaje funcionan correctamente")
        except Exception as e:
            print(f"✗ Error en curvas de aprendizaje: {e}")
        
        # Probar matriz de confusión
        print("\nProbando matriz de confusión...")
        try:
            confusion = evaluator.plot_confusion_matrix(model, X_test, y_test, return_data=True)
            print(f"- Matriz de confusión generada")
            print("✓ Matriz de confusión funciona correctamente")
        except Exception as e:
            print(f"✗ Error en matriz de confusión: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Error al importar módulos: {e}")
        return False
    except Exception as e:
        print(f"✗ Error general: {e}")
        return False

def check_api_service():
    """Verifica la configuración del servicio API"""
    print("\n=== VERIFICANDO SERVICIO API ===")
    
    api_path = os.path.join(script_dir, 'ml_service_v2.py')
    if not os.path.exists(api_path):
        print(f"✗ No se encontró el archivo del servicio API en {api_path}")
        return False
        
    print(f"✓ Archivo de servicio API encontrado")
    
    try:
        # Verificar que los endpoints necesarios están implementados
        with open(api_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_endpoints = [
            '/status', 
            '/models/available', 
            '/models/train', 
            '/clustering/perform',
            '/timeseries/analyze', 
            '/models/evaluate'
        ]
        
        missing = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing.append(endpoint)
        
        if missing:
            print(f"✗ Faltan los siguientes endpoints: {', '.join(missing)}")
        else:
            print(f"✓ Todos los endpoints requeridos están implementados")
        
        # Verificar importaciones de módulos avanzados
        required_imports = [
            'from advanced_models import',
            'from advanced_clustering import',
            'from time_series_analyzer import',
            'from model_evaluator import'
        ]
        
        missing_imports = []
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"✗ Faltan las siguientes importaciones: {', '.join(missing_imports)}")
        else:
            print(f"✓ Todas las importaciones necesarias están presentes")
            
        return len(missing) == 0 and len(missing_imports) == 0
    except Exception as e:
        print(f"✗ Error al verificar servicio API: {e}")
        return False

if __name__ == "__main__":
    print("=== PRUEBA DE MODELOS AVANZADOS DE MACHINE LEARNING ===")
    
    # Ejecutar todas las pruebas
    results = {
        "Modelos Avanzados": test_advanced_models(),
        "Clustering": test_clustering(),
        "Series Temporales": test_time_series(),
        "Evaluación de Modelos": test_model_evaluation(),
        "API Service": check_api_service()
    }
    
    # Mostrar resultados
    print("\n=== RESUMEN DE RESULTADOS ===")
    all_pass = True
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ TODOS LOS COMPONENTES FUNCIONAN CORRECTAMENTE")
        print("\nPuede iniciar el servicio API con:")
        print("python data_science_ml_learning/ml_service_v2.py")
        print("\nY acceder a la documentación de la API en:")
        print("http://localhost:8000/docs")
    else:
        print("\n✗ ALGUNOS COMPONENTES NECESITAN CORRECCIÓN")
        print("Revise los errores específicos para cada componente que falló.")
