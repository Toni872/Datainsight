#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modelos Avanzados de Machine Learning
Este módulo implementa algoritmos avanzados de clasificación y regresión
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import time
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# Intentar importar xgboost y lightgbm (si están disponibles)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost no está instalado. Para usar XGBoost, instale: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM no está instalado. Para usar LightGBM, instale: pip install lightgbm")

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AdvancedModelTrainer:
    """Clase para entrenar modelos avanzados de machine learning"""
    
    def __init__(self):
        """Inicializa el entrenador con modelos avanzados disponibles"""
        # Modelos de clasificación disponibles
        self.classification_models = {
            'gradient_boosting': GradientBoostingClassifier,
            'mlp': MLPClassifier,
            'ada_boost': AdaBoostClassifier,
            'stacking': self._get_stacking_classifier  # Añadido modelo de stacking
        }
        
        # Modelos de regresión disponibles
        self.regression_models = {
            'gradient_boosting': GradientBoostingRegressor,
            'mlp': MLPRegressor,
            'ada_boost': AdaBoostRegressor,
            'stacking': self._get_stacking_regressor  # Añadido modelo de stacking
        }
        
        # Añadir xgboost si está disponible
        if XGBOOST_AVAILABLE:
            self.classification_models['xgboost'] = xgb.XGBClassifier
            self.regression_models['xgboost'] = xgb.XGBRegressor
            
        # Añadir lightgbm si está disponible
        if LIGHTGBM_AVAILABLE:
            self.classification_models['lightgbm'] = lgb.LGBMClassifier
            self.regression_models['lightgbm'] = lgb.LGBMRegressor
        
        # Parámetros predeterminados para ajuste de hiperparámetros
        self.hyperparameters = {
            'gradient_boosting_classifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'mlp_classifier': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'xgboost_classifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm_classifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, -1],
                'num_leaves': [31, 50, 100]
            }
            # Los parámetros para modelos de regresión son similares
        }
    def _get_stacking_classifier(self, **kwargs):
        """
        Crea un modelo de stacking para clasificación
        
        Args:
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            StackingClassifier: Modelo de stacking para clasificación
        """
        # Definir estimadores base
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        # Si XGBoost está disponible, añadirlo como estimador
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)))
        
        # Crear modelo de stacking con un meta-clasificador logístico
        from sklearn.linear_model import LogisticRegression
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method='predict_proba'
        )
        
        # Actualizar parámetros con los proporcionados
        if kwargs:
            for param, value in kwargs.items():
                if hasattr(model, param):
                    setattr(model, param, value)
        
        return model
        
    def _get_stacking_regressor(self, **kwargs):
        """
        Crea un modelo de stacking para regresión
        
        Args:
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            StackingRegressor: Modelo de stacking para regresión
        """
        from sklearn.svm import SVR
        
        # Definir estimadores base
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('svr', SVR(kernel='rbf'))
        ]
        
        # Si XGBoost está disponible, añadirlo como estimador
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)))
        
        # Crear modelo de stacking con un meta-regresor Ridge
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=42),
            cv=5
        )
        
        # Actualizar parámetros con los proporcionados
        if kwargs:
            for param, value in kwargs.items():
                if hasattr(model, param):
                    setattr(model, param, value)
        
        return model

    def get_available_models(self, model_type='classification'):
        """
        Devuelve los modelos disponibles para el tipo especificado
        
        Args:
            model_type (str): 'classification' o 'regression'
            
        Returns:
            dict: Diccionario con los modelos disponibles
        """
        if model_type == 'classification':
            return self.classification_models
        elif model_type == 'regression':
            return self.regression_models
        else:
            raise ValueError(f"Tipo de modelo no válido: {model_type}")
    
    def create_stacking_model(self, base_models, model_type='classification', final_estimator=None):
        """
        Crea un modelo de stacking (ensamble) con los modelos base especificados
        
        Args:
            base_models (list): Lista de tuplas (nombre, modelo) para usar como base
            model_type (str): 'classification' o 'regression'
            final_estimator: Modelo final para la capa de meta-aprendizaje
            
        Returns:
            estimator: Modelo de stacking configurado
        """
        if model_type == 'classification':
            if final_estimator is None:
                final_estimator = LogisticRegression()
            return StackingClassifier(estimators=base_models, final_estimator=final_estimator)
        elif model_type == 'regression':
            if final_estimator is None:
                final_estimator = LinearRegression()
            return StackingRegressor(estimators=base_models, final_estimator=final_estimator)
        else:
            raise ValueError(f"Tipo de modelo no válido: {model_type}")
    
    def create_pipeline_with_feature_engineering(self, estimator, include_pca=False, 
                                                poly_degree=None, scale_data=True):
        """
        Crea un pipeline con pasos de ingeniería de características
        
        Args:
            estimator: Modelo de ML a utilizar al final del pipeline
            include_pca (bool): Si se debe incluir PCA para reducción de dimensionalidad
            poly_degree (int): Grado para añadir características polinómicas (None para omitir)
            scale_data (bool): Si se deben escalar los datos
            
        Returns:
            Pipeline: Pipeline de scikit-learn configurado
        """
        steps = []
        
        # Añadir escalado si se solicita
        if scale_data:
            steps.append(('scaler', StandardScaler()))
        
        # Añadir características polinómicas si se especifica grado
        if poly_degree is not None and poly_degree > 1:
            steps.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)))
            
            # Si agregamos poly features, es buena idea escalar de nuevo
            if not scale_data:
                steps.append(('scaler_after_poly', StandardScaler()))
        
        # Añadir PCA si se solicita
        if include_pca:
            # Por defecto, mantener el 95% de la varianza
            steps.append(('pca', PCA(n_components=0.95)))
        
        # Añadir el modelo final
        steps.append(('model', estimator))
        
        return Pipeline(steps=steps)
    
    def tune_hyperparameters(self, model_name, model, X_train, y_train, param_grid=None, 
                           cv=5, scoring=None, n_jobs=-1, search_type='grid'):
        """
        Realiza ajuste de hiperparámetros para un modelo
        
        Args:
            model_name (str): Nombre del modelo
            model: Instancia del modelo a optimizar
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            param_grid (dict): Rejilla de parámetros para búsqueda
            cv (int): Número de folds para validación cruzada
            scoring (str): Métrica para optimización
            n_jobs (int): Número de trabajos paralelos
            search_type (str): 'grid' para GridSearchCV o 'random' para RandomizedSearchCV
            
        Returns:
            estimator: Modelo optimizado
        """
        # Si no se proporciona param_grid, usar preconfigurados
        if param_grid is None:
            # Intentar obtener del diccionario de hiperparámetros predefinidos
            param_key = f"{model_name}"
            if param_key in self.hyperparameters:
                param_grid = self.hyperparameters[param_key]
            else:
                # Si no hay parámetros predefinidos, usar un conjunto mínimo
                print(f"No se encontraron hiperparámetros predefinidos para {model_name}")
                return model  # Devolver el modelo sin optimizar
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
            )
        else:  # Random search
            # Para random search, es mejor tener más iteraciones
            n_iter = 20 if len(param_grid) > 3 else 10
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, 
                scoring=scoring, n_jobs=n_jobs, verbose=1
            )
        
        search.fit(X_train, y_train)
        print(f"Mejores parámetros para {model_name}: {search.best_params_}")
        print(f"Mejor puntuación CV: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train_model_with_evaluation(self, model_name, model_type, X_train, y_train, X_test, y_test, 
                                  feature_names=None, class_names=None, optimize=True):
        """
        Entrena un modelo avanzado y proporciona evaluación detallada
        
        Args:
            model_name (str): Nombre del modelo a entrenar
            model_type (str): 'classification' o 'regression'
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            feature_names: Nombres de las características
            class_names: Nombres de las clases (para clasificación)
            optimize (bool): Si se debe realizar optimización de hiperparámetros
            
        Returns:
            dict: Resultados del entrenamiento y evaluación
        """
        # Seleccionar el tipo de modelo
        if model_type == 'classification':
            models_dict = self.classification_models
            # Para clasificación binaria o multiclase
            is_binary = len(np.unique(y_train)) == 2
            scoring = 'f1' if is_binary else 'f1_weighted'
        elif model_type == 'regression':
            models_dict = self.regression_models
            scoring = 'neg_mean_squared_error'
        else:
            raise ValueError(f"Tipo de modelo no válido: {model_type}")
        
        # Verificar si el modelo solicitado está disponible
        if model_name not in models_dict:
            available = list(models_dict.keys())
            raise ValueError(f"Modelo {model_name} no disponible. Opciones: {available}")
            
        # Crear modelo
        model_class = models_dict[model_name]
        model = model_class()
        
        # Optimizar hiperparámetros si se solicita
        if optimize:
            model = self.tune_hyperparameters(
                f"{model_name}_{model_type}", model, X_train, y_train, 
                scoring=scoring, search_type='random'
            )
        
        # Entrenar modelo
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluar modelo
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'train_time': train_time,
            'feature_names': feature_names if feature_names else [],
            'parameters': model.get_params()
        }
        
        # Métricas específicas según el tipo de modelo
        if model_type == 'classification':
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Probabilidades si el modelo lo soporta
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                results['prediction_probabilities_available'] = True
            else:
                results['prediction_probabilities_available'] = False
            
            # Métricas de clasificación
            results.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'class_names': class_names if class_names else [str(i) for i in np.unique(y_train)]
            })
            
            # Feature importance si está disponible
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = {
                    feature_names[i] if feature_names else f"feature_{i}": float(imp)
                    for i, imp in enumerate(model.feature_importances_)
                }
            
        elif model_type == 'regression':
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas de regresión
            results.update({
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred),
            })
            
            # Feature importance si está disponible
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = {
                    feature_names[i] if feature_names else f"feature_{i}": float(imp)
                    for i, imp in enumerate(model.feature_importances_)
                }
        
        return results

    def train(self, X, y, algorithm, task_type, **kwargs):
        """
        Entrena un modelo avanzado con los datos proporcionados
        
        Args:
            X (array): Características de entrenamiento
            y (array): Etiquetas o valores objetivo
            algorithm (str): Nombre del algoritmo a utilizar
            task_type (str): 'classification' o 'regression'
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            dict: Resultados del entrenamiento, incluyendo métricas y modelo entrenado
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"task_type debe ser 'classification' o 'regression', se recibió: {task_type}")
        
        # Seleccionar los modelos disponibles según el tipo de tarea
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        if algorithm not in models:
            available = list(models.keys())
            raise ValueError(f"Algoritmo '{algorithm}' no disponible. Opciones: {available}")
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Aplicar preprocesamiento
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
          # Crear el modelo
        if algorithm == 'gradient_boosting':
            if task_type == 'classification':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif algorithm == 'mlp':
            if task_type == 'classification':
                model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            else:
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        elif algorithm == 'stacking':
            if task_type == 'classification':
                model = self._get_stacking_classifier()
            else:
                model = self._get_stacking_regressor()
        elif algorithm == 'ada_boost':
            if task_type == 'classification':
                model = AdaBoostClassifier(random_state=42)
            else:
                model = AdaBoostRegressor(random_state=42)
        elif XGBOOST_AVAILABLE and algorithm == 'xgboost':
            if task_type == 'classification':
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif LIGHTGBM_AVAILABLE and algorithm == 'lightgbm':
            if task_type == 'classification':
                model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            else:
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            # Instanciar cualquier otro modelo disponible en el diccionario
            model = models[algorithm]()
        
        # Actualizar parámetros del modelo con los proporcionados
        if kwargs:
            model.set_params(**kwargs)
        
        # Entrenar modelo
        model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        if task_type == 'classification':
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            result = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix
            }
        else:
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result = {
                'model': model,
                'scaler': scaler,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        
        return result
        
    def predict(self, model_info, X):
        """
        Realiza predicciones con un modelo entrenado
        
        Args:
            model_info (dict): Diccionario con el modelo y el scaler
            X (array): Datos para predecir
            
        Returns:
            array: Predicciones
        """
        if isinstance(model_info, dict):
            model = model_info['model']
            scaler = model_info['scaler']
            X_scaled = scaler.transform(X)
        else:
            # Asumir que model_info es el modelo directamente
            model = model_info
            X_scaled = X
            
        return model.predict(X_scaled)
        
    def train_with_hyperparameter_tuning(self, X, y, algorithm, task_type, cv=5, n_iter=20):
        """
        Entrena un modelo con optimización de hiperparámetros
        
        Args:
            X (array): Características de entrenamiento
            y (array): Etiquetas o valores objetivo
            algorithm (str): Nombre del algoritmo a utilizar
            task_type (str): 'classification' o 'regression'
            cv (int): Número de folds para validación cruzada
            n_iter (int): Número de iteraciones para RandomizedSearchCV
            
        Returns:
            dict: Resultados del entrenamiento con optimización
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"task_type debe ser 'classification' o 'regression', se recibió: {task_type}")
        
        # Seleccionar los modelos disponibles según el tipo de tarea
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        if algorithm not in models:
            available = list(models.keys())
            raise ValueError(f"Algoritmo '{algorithm}' no disponible. Opciones: {available}")
        
        # Obtener el grid de parámetros para el algoritmo
        param_grid = self.get_default_param_grid(algorithm)
        
        if not param_grid:
            raise ValueError(f"No se encontró grid de parámetros para {algorithm}")
        
        # Aplicar preprocesamiento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear el modelo base
        if algorithm == 'gradient_boosting':
            if task_type == 'classification':
                model = GradientBoostingClassifier()
            else:
                model = GradientBoostingRegressor()
        elif algorithm == 'mlp':
            if task_type == 'classification':
                model = MLPClassifier()
            else:
                model = MLPRegressor()
        elif XGBOOST_AVAILABLE and algorithm == 'xgboost':
            if task_type == 'classification':
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
        elif LIGHTGBM_AVAILABLE and algorithm == 'lightgbm':
            if task_type == 'classification':
                model = lgb.LGBMClassifier()
            else:
                model = lgb.LGBMRegressor()
        else:
            # Instanciar cualquier otro modelo disponible
            model = models[algorithm]()
        
        # Realizar búsqueda de hiperparámetros
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy' if task_type == 'classification' else 'r2',
            random_state=42, n_jobs=-1
        )
        
        search.fit(X_scaled, y)
        
        # Obtener el mejor modelo
        best_model = search.best_estimator_
        
        # Evaluar el mejor modelo
        from sklearn.model_selection import cross_val_score
        if task_type == 'classification':
            scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='f1_weighted')
            
            result = {
                'model': best_model,
                'scaler': scaler,
                'best_params': search.best_params_,
                'accuracy': search.best_score_,
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_f1_mean': f1_scores.mean(),
                'cv_f1_std': f1_scores.std()
            }
        else:
            scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='r2')
            mse_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
            
            result = {
                'model': best_model,
                'scaler': scaler,
                'best_params': search.best_params_,
                'r2': search.best_score_,
                'cv_r2_mean': scores.mean(),
                'cv_r2_std': scores.std(),
                'cv_mse_mean': -mse_scores.mean(),
                'cv_rmse_mean': np.sqrt(-mse_scores.mean())
            }
        
        return result
        
    def get_default_param_grid(self, algorithm):
        """
        Obtiene el grid de parámetros por defecto para un algoritmo
        
        Args:
            algorithm (str): Nombre del algoritmo
            
        Returns:
            dict: Grid de parámetros para el algoritmo
        """
        param_grids = {
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['linear', 'rbf', 'poly']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['linear', 'rbf', 'poly']
            }
        }
        
        return param_grids.get(algorithm, None)

# Prueba del módulo si se ejecuta directamente
if __name__ == '__main__':
    print("Módulo de modelos avanzados de Machine Learning")
    
    # Mostrar modelos disponibles
    trainer = AdvancedModelTrainer()
    print("\nModelos de clasificación disponibles:")
    for name in trainer.classification_models.keys():
        print(f"- {name}")
    
    print("\nModelos de regresión disponibles:")
    for name in trainer.regression_models.keys():
        print(f"- {name}")
