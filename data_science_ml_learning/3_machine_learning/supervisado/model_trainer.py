#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Trainer - Módulo para entrenar y evaluar modelos de machine learning
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelTrainer:
    """Clase para entrenar y evaluar modelos de machine learning"""
    
    def __init__(self):
        """Inicializa el entrenador de modelos"""
        self.models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.models_path, exist_ok=True)
        
        # Mapear nombres de modelos a sus clases
        self.classification_models = {
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression,
            'knn': KNeighborsClassifier,
            'decision_tree': DecisionTreeClassifier
        }
        
        self.regression_models = {
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'svr': SVR,
            'random_forest_regressor': RandomForestRegressor
        }
        
        self.clustering_models = {
            'kmeans': KMeans,
            'dbscan': DBSCAN
        }
    
    def train_models(self, data_file, model_type, models_to_train, cv_folds=5):
        """
        Entrena múltiples modelos en los datos proporcionados
        
        Args:
            data_file (str): Ruta al archivo de datos JSON
            model_type (str): Tipo de modelo ('classification', 'regression', 'clustering')
            models_to_train (list): Lista de nombres de modelos a entrenar
            cv_folds (int): Número de folds para validación cruzada
            
        Returns:
            dict: Resultados del entrenamiento
        """
        # Cargar datos
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            X_train = np.array(data['X_train'])
            X_test = np.array(data['X_test'])
            y_train = np.array(data['y_train'])
            y_test = np.array(data['y_test'])
            feature_names = data['feature_names']
            dataset_id = data['dataset_id']
        except Exception as e:
            return {'error': f'Error al cargar datos: {str(e)}'}
        
        # Seleccionar modelos basados en el tipo
        if model_type == 'classification':
            models_dict = self.classification_models
        elif model_type == 'regression':
            models_dict = self.regression_models
        elif model_type == 'clustering':
            models_dict = self.clustering_models
        else:
            return {'error': f'Tipo de modelo no válido: {model_type}'}
        
        # Filtrar solo los modelos solicitados
        filtered_models = {name: cls for name, cls in models_dict.items() if name in models_to_train}
        
        if not filtered_models:
            return {'error': 'Ninguno de los modelos solicitados es válido'}
        
        # Resultados
        results = {
            'models_performance': {},
            'training_time': {},
            'best_model': None,
            'best_score': 0 if model_type != 'regression' else float('inf'),
            'feature_importance': {},
            'confusion_matrices': {},
            'roc_curves': {},
            'dataset_id': dataset_id,
            'model_type': model_type,
            'models_trained': list(filtered_models.keys())
        }
        
        # Entrenamiento y evaluación de cada modelo
        for name, model_class in filtered_models.items():
            start_time = time.time()
            
            try:
                # Inicializar modelo con parámetros predeterminados
                if name == 'svm' or name == 'svr':
                    model = model_class(probability=True) if name == 'svm' else model_class()
                else:
                    model = model_class()
                
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Tiempo de entrenamiento
                train_time = time.time() - start_time
                results['training_time'][name] = train_time
                
                # Guardar modelo entrenado
                model_file = os.path.join(self.models_path, f"{dataset_id}_{name}.joblib")
                dump(model, model_file)
                
                # Evaluación según el tipo de modelo
                if model_type == 'classification':
                    # Predicciones
                    y_pred = model.predict(X_test)
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                    else:
                        y_prob = None
                    
                    # Métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Validación cruzada
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                    
                    # Matriz de confusión
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Guardar resultados
                    results['models_performance'][name] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'cv_scores': cv_scores.tolist(),
                        'cv_mean': float(cv_scores.mean()),
                        'cv_std': float(cv_scores.std())
                    }
                    
                    results['confusion_matrices'][name] = cm.tolist()
                    
                    # Curva ROC para clasificación binaria
                    if len(np.unique(y_test)) == 2 and y_prob is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        roc_auc = auc(fpr, tpr)
                        results['roc_curves'][name] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(roc_auc)
                        }
                    
                    # Determinar el mejor modelo por F1
                    if results['models_performance'][name]['f1_score'] > results['best_score']:
                        results['best_model'] = name
                        results['best_score'] = results['models_performance'][name]['f1_score']
                    
                elif model_type == 'regression':
                    # Predicciones
                    y_pred = model.predict(X_test)
                    
                    # Métricas
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Validación cruzada
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
                    
                    # Guardar resultados
                    results['models_performance'][name] = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2_score': float(r2),
                        'mae': float(mae),
                        'cv_scores': (-cv_scores).tolist(),
                        'cv_mean': float(-cv_scores.mean()),
                        'cv_std': float(cv_scores.std())
                    }
                    
                    # Determinar el mejor modelo por RMSE (menor es mejor)
                    if results['models_performance'][name]['rmse'] < results['best_score']:
                        results['best_model'] = name
                        results['best_score'] = results['models_performance'][name]['rmse']
                
                elif model_type == 'clustering':
                    # Para clustering, evaluamos con métricas diferentes (silhouette score)
                    from sklearn.metrics import silhouette_score
                    
                    # Predecir clusters
                    if name == 'dbscan':
                        # DBSCAN no necesita predecir en datos de prueba, usa los mismos clusters
                        y_pred = model.fit_predict(X_test)
                        # DBSCAN puede devolver -1 para ruido, excluimos estos puntos para la métrica silhouette
                        mask = y_pred != -1
                        if mask.any():  # Solo calcular si hay puntos no ruidosos
                            silhouette = silhouette_score(X_test[mask], y_pred[mask]) if len(np.unique(y_pred[mask])) > 1 else 0
                        else:
                            silhouette = 0
                    else:  # KMeans
                        y_pred = model.predict(X_test)
                        silhouette = silhouette_score(X_test, y_pred) if len(np.unique(y_pred)) > 1 else 0
                    
                    # Número de clusters
                    n_clusters = len(np.unique(y_pred))
                    if name == 'dbscan' and -1 in y_pred:
                        n_clusters -= 1  # No contar el ruido como un cluster
                    
                    # Guardar resultados
                    results['models_performance'][name] = {
                        'silhouette_score': float(silhouette),
                        'n_clusters': int(n_clusters)
                    }
                    
                    # En clustering, usamos silhouette score como métrica (mayor es mejor)
                    if results['models_performance'][name]['silhouette_score'] > results['best_score']:
                        results['best_model'] = name
                        results['best_score'] = results['models_performance'][name]['silhouette_score']
                
                # Importancia de características (solo para modelos que la soportan)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_imp = dict(zip(feature_names, importances.tolist()))
                    # Ordenar por importancia
                    feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}
                    results['feature_importance'][name] = feature_imp
                elif hasattr(model, 'coef_'):
                    # Para modelos lineales
                    coefs = model.coef_
                    if coefs.ndim > 1:  # Para clasificadores multiclase
                        coefs = np.abs(coefs).mean(axis=0)
                    feature_imp = dict(zip(feature_names, coefs.tolist()))
                    # Ordenar por importancia absoluta
                    feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: abs(item[1]), reverse=True)}
                    results['feature_importance'][name] = feature_imp
                
            except Exception as e:
                results['models_performance'][name] = {'error': str(e)}
        
        # Si es clustering, generar visualización
        if model_type == 'clustering' and results['best_model'] and X_test.shape[1] > 2:
            try:
                # Usar PCA para visualizar clusters en 2D
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_test)
                
                best_model_name = results['best_model']
                best_model = load(os.path.join(self.models_path, f"{dataset_id}_{best_model_name}.joblib"))
                
                if best_model_name == 'dbscan':
                    y_pred = best_model.fit_predict(X_test)
                else:
                    y_pred = best_model.predict(X_test)
                
                # Crear gráfico
                plt.figure(figsize=(10, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
                plt.title(f'Clusters usando {best_model_name.upper()} (PCA)')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.colorbar(label='Cluster')
                
                # Guardar gráfico
                plot_file = os.path.join(self.models_path, f"{dataset_id}_{best_model_name}_clusters.png")
                plt.savefig(plot_file)
                plt.close()
                
                results['clustering_plot'] = plot_file
                
            except Exception as e:
                results['clustering_plot_error'] = str(e)
        
        return results
    
    def predict(self, model_name, dataset_id, features):
        """
        Realiza una predicción con un modelo entrenado
        
        Args:
            model_name (str): Nombre del modelo
            dataset_id (str): ID del dataset
            features (list): Lista de valores de características
            
        Returns:
            dict: Resultado de la predicción
        """
        try:
            # Cargar el modelo
            model_file = os.path.join(self.models_path, f"{dataset_id}_{model_name}.joblib")
            
            if not os.path.exists(model_file):
                return {'error': f'Modelo {model_name} para dataset {dataset_id} no encontrado'}
            
            model = load(model_file)
            
            # Convertir características a array numpy
            X = np.array(features).reshape(1, -1)
            
            # Realizar predicción
            prediction = model.predict(X)[0]
            
            result = {
                'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else float(prediction)
            }
            
            # Si es un modelo de clasificación con probabilidades
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                
                # Si hay clases definidas en el modelo
                if hasattr(model, 'classes_'):
                    classes = model.classes_
                    probs_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
                    result['probabilities'] = probs_dict
                else:
                    result['probabilities'] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            return {'error': f'Error al realizar predicción: {str(e)}'}


# Función principal para usar desde línea de comandos
def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Se requiere una acción'}))
        return
    
    action = sys.argv[1]
    trainer = ModelTrainer()
    
    if action == 'train':
        if len(sys.argv) < 5:
            print(json.dumps({'error': 'Se requieren archivo de datos, tipo de modelo y modelos a entrenar'}))
            return
        
        data_file = sys.argv[2]
        model_type = sys.argv[3]
        models_to_train = sys.argv[4].split(',')
        cv_folds = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        
        results = trainer.train_models(data_file, model_type, models_to_train, cv_folds)
        print(json.dumps(results))
    
    elif action == 'predict':
        if len(sys.argv) < 5:
            print(json.dumps({'error': 'Se requiere nombre de modelo, dataset_id y características'}))
            return
        
        model_name = sys.argv[2]
        dataset_id = sys.argv[3]
        features = json.loads(sys.argv[4])
        
        result = trainer.predict(model_name, dataset_id, features)
        print(json.dumps(result))
    
    else:
        print(json.dumps({'error': 'Acción no reconocida'}))


if __name__ == '__main__':
    main()