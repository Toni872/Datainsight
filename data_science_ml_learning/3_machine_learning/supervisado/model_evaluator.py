#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades de Evaluación Avanzada de Modelos
Este módulo proporciona herramientas para evaluación avanzada de modelos ML
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from joblib import Parallel, delayed
from sklearn.model_selection import (
    learning_curve, validation_curve, KFold, StratifiedKFold,
    cross_val_score, cross_validate
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, mean_squared_error,
    r2_score, mean_absolute_error
)
from sklearn.calibration import calibration_curve

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelEvaluator:
    """Clase para evaluación avanzada de modelos de machine learning"""
    
    def __init__(self):
        """Inicializa el evaluador de modelos"""
        pass
    
    def generate_learning_curves(self, estimator, X, y, cv=5, n_jobs=-1, 
                                train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        Genera curvas de aprendizaje para un modelo
        
        Args:
            estimator: Modelo a evaluar
            X: Datos de características
            y: Datos de etiquetas
            cv (int): Número de folds para validación cruzada
            n_jobs (int): Número de trabajos paralelos
            train_sizes: Tamaños relativos para entrenamiento
            
        Returns:
            dict: Resultados de curvas de aprendizaje y figura como buffer
        """
        # Configurar validación cruzada apropiadamente
        if hasattr(y, 'dtype') and y.dtype.kind in 'if':  # Regresión
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        else:  # Clasificación
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Calcular curvas de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv_obj, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring='r2' if y.dtype.kind in 'if' else 'accuracy'
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        
        ax.plot(train_sizes, test_mean, 'o-', color='green', label='Validación')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='green')
        
        ax.set_title('Curvas de Aprendizaje')
        ax.set_xlabel('Tamaño de datos de entrenamiento')
        ax.set_ylabel('Puntuación')
        ax.grid(True)
        ax.legend(loc='best')
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Preparar resultados
        results = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'test_scores_mean': test_mean.tolist(),
            'test_scores_std': test_std.tolist()
        }
        
        return results, buf
    
    def generate_validation_curves(self, estimator, X, y, param_name, param_range, cv=5, n_jobs=-1):
        """
        Genera curvas de validación para un parámetro de un modelo
        
        Args:
            estimator: Modelo a evaluar
            X: Datos de características
            y: Datos de etiquetas
            param_name (str): Nombre del parámetro a evaluar
            param_range (list): Rango de valores para el parámetro
            cv (int): Número de folds para validación cruzada
            n_jobs (int): Número de trabajos paralelos
            
        Returns:
            dict: Resultados de curvas de validación y figura como buffer
        """
        # Configurar validación cruzada apropiadamente
        if hasattr(y, 'dtype') and y.dtype.kind in 'if':  # Regresión
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
        else:  # Clasificación
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'accuracy'
        
        # Calcular curvas de validación
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv_obj, n_jobs=n_jobs, scoring=scoring
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Transformar MSE negativo a positivo para regresión
        if scoring == 'neg_mean_squared_error':
            train_mean = -train_mean
            test_mean = -test_mean
            ylabel = 'MSE'
        else:
            ylabel = 'Precisión'
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_range, train_mean, 'o-', color='blue', label='Entrenamiento')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        
        ax.plot(param_range, test_mean, 'o-', color='green', label='Validación')
        ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='green')
        
        ax.set_title(f'Curvas de Validación para {param_name}')
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc='best')
        
        # Configurar escala logarítmica si el rango lo requiere
        if max(param_range) / min(param_range) > 100:
            ax.set_xscale('log')
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Preparar resultados
        results = {
            'param_name': param_name,
            'param_range': [str(p) for p in param_range],  # Convertir a string para serialización JSON
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'test_scores_mean': test_mean.tolist(),
            'test_scores_std': test_std.tolist()
        }
        
        return results, buf
    
    def evaluate_feature_importance(self, estimator, X, y, feature_names=None, n_repeats=10):
        """
        Evalúa la importancia de características usando permutaciones
        
        Args:
            estimator: Modelo entrenado
            X: Datos de características
            y: Datos de etiquetas
            feature_names (list): Nombres de las características
            n_repeats (int): Número de repeticiones para la evaluación
            
        Returns:
            dict: Resultados de importancia de características y figura como buffer
        """
        # Determinar si es clasificación o regresión
        if hasattr(y, 'dtype') and y.dtype.kind in 'if':  # Regresión
            scoring = 'r2'
        else:  # Clasificación
            scoring = 'accuracy'
        
        # Calcular importancia de características
        result = permutation_importance(
            estimator, X, y, n_repeats=n_repeats, 
            random_state=42, scoring=scoring
        )
        
        # Ordenar características por importancia
        perm_sorted_idx = result.importances_mean.argsort()[::-1]
        
        # Usar nombres de características si están disponibles
        if feature_names is None:
            feature_names = [f'Caracteristica {i+1}' for i in range(X.shape[1])]
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sorted_idx = perm_sorted_idx[:15]  # Limitar a las 15 más importantes para claridad
        y_pos = np.arange(len(sorted_idx))
        
        ax.barh(
            y_pos, 
            result.importances_mean[sorted_idx],
            xerr=result.importances_std[sorted_idx],
            align='center',
            alpha=0.8,
            color='skyblue'
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Importancia')
        ax.set_title('Importancia de Características (Permutación)')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Preparar resultados completos
        all_importances = {
            feature_names[i]: {
                'importance_mean': float(result.importances_mean[i]),
                'importance_std': float(result.importances_std[i])
            }
            for i in range(len(feature_names))
        }
        
        # Ordenar por importancia
        sorted_importances = dict(sorted(
            all_importances.items(),
            key=lambda item: item[1]['importance_mean'],
            reverse=True
        ))
        
        return sorted_importances, buf
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names=None):
        """
        Genera curvas ROC para clasificación
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            class_names (list): Nombres de las clases
            
        Returns:
            dict: Resultados y figura como buffer
        """
        # Convertir a array numpy si no lo es
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Manejar caso binario
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
            
        # Preparar curva ROC
        if y_pred_proba.ndim == 1:  # Binaria
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Crear gráfica
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(
                fpr, tpr, color='darkorange', lw=2,
                label=f'Área ROC = {roc_auc:.3f}'
            )
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Preparar resultados
            results = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }
            
        else:  # Multiclase
            # Configurar gráfica
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Si no hay nombres de clase, crear genéricos
            if class_names is None:
                class_names = [f'Clase {i}' for i in range(y_pred_proba.shape[1])]
            
            results = {'classes': {}}
            
            # Calcular ROC para cada clase
            for i, class_name in enumerate(class_names):
                # Convertir a one-vs-rest
                y_true_bin = (y_true == i).astype(int)
                y_score = y_pred_proba[:, i]
                
                fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Graficar curva ROC para la clase
                ax.plot(
                    fpr, tpr, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})'
                )
                
                # Guardar resultados
                results['classes'][class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
            
            # Finalizar gráfico
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curvas ROC Multiclase (One-vs-Rest)')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return results, buf
    
    def evaluate_model_cross_validation(self, estimator, X, y, cv=5, scoring=None, n_jobs=-1):
        """
        Evalúa un modelo usando validación cruzada con múltiples métricas
        
        Args:
            estimator: Modelo a evaluar
            X: Datos de características
            y: Datos de etiquetas
            cv (int): Número de folds para validación cruzada
            scoring (dict/list): Métricas para evaluación
            n_jobs (int): Número de trabajos paralelos
            
        Returns:
            dict: Resultados de validación cruzada
        """
        # Determinar tipo de problema y configurar métricas
        is_regression = hasattr(y, 'dtype') and y.dtype.kind in 'if'
        
        if scoring is None:
            if is_regression:
                scoring = ['neg_mean_squared_error', 'r2']
            else:
                # Detectar si es multiclase o binaria
                n_classes = len(np.unique(y))
                if n_classes <= 2:
                    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                else:
                    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # Configurar validación cruzada apropiadamente
        if is_regression:
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Realizar validación cruzada
        cv_results = cross_validate(
            estimator, X, y, cv=cv_obj, scoring=scoring, 
            n_jobs=n_jobs, return_train_score=True
        )
        
        # Calcular medias y desviaciones estándar
        results = {}
        
        for metric in cv_results.keys():
            if metric.startswith('test_') or metric.startswith('train_'):
                base_name = metric.split('_', 1)[1]
                group = metric.split('_', 1)[0]  # 'test' o 'train'
                
                if group not in results:
                    results[group] = {}
                
                # Convertir nombres de métricas negativas
                display_name = base_name
                if base_name.startswith('neg_'):
                    display_name = base_name[4:]
                    cv_results[metric] = -cv_results[metric]  # Convertir a positivo
                
                results[group][display_name] = {
                    'mean': float(np.mean(cv_results[metric])),
                    'std': float(np.std(cv_results[metric])),
                    'values': [float(v) for v in cv_results[metric]]
                }
        
        # Añadir tiempo de ajuste y puntuación
        results['fit_time'] = {
            'mean': float(np.mean(cv_results['fit_time'])),
            'std': float(np.std(cv_results['fit_time'])),
            'values': [float(t) for t in cv_results['fit_time']]
        }
        
        results['score_time'] = {
            'mean': float(np.mean(cv_results['score_time'])),
            'std': float(np.std(cv_results['score_time'])),
            'values': [float(t) for t in cv_results['score_time']]
        }
        
        return results

    def evaluate_model(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Realiza una evaluación completa de un modelo de machine learning
        
        Args:
            model: Modelo entrenado para evaluar
            X_test: Características del conjunto de prueba
            y_test: Etiquetas del conjunto de prueba
            X_train: Características del conjunto de entrenamiento (opcional)
            y_train: Etiquetas del conjunto de entrenamiento (opcional)
            
        Returns:
            dict: Métricas de evaluación y visualizaciones
        """
        results = {}
        
        # Determinar si es un problema de clasificación o regresión
        # Si hay sólo valores 0 y 1 en y_test o solo unas pocas clases discretas
        unique_labels = np.unique(y_test)
        is_classification = len(unique_labels) < 20 and np.issubdtype(y_test.dtype, np.integer)
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Añadir métricas básicas
        if is_classification:
            # Es un problema de clasificación
            results['accuracy'] = model.score(X_test, y_test)
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Para modelos que pueden dar probabilidades
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    
                    # Para clasificación binaria
                    if len(unique_labels) == 2:
                        # Curvas ROC
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        results['roc'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(auc(fpr, tpr))
                        }
                        
                        # Curvas Precision-Recall
                        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                        results['pr_curve'] = {
                            'precision': precision.tolist(),
                            'recall': recall.tolist(),
                            'average_precision': float(average_precision_score(y_test, y_prob[:, 1]))
                        }
                        
                        # Curva de calibración
                        fraction_positive, mean_predicted_value = calibration_curve(
                            y_test, y_prob[:, 1], n_bins=10
                        )
                        results['calibration'] = {
                            'fraction_positive': fraction_positive.tolist(),
                            'mean_predicted_value': mean_predicted_value.tolist()
                        }
                except:
                    # Algunos modelos pueden no soportar predict_proba correctamente
                    pass
                
        else:
            # Es un problema de regresión
            results['r2_score'] = r2_score(y_test, y_pred)
            results['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            results['root_mean_squared_error'] = np.sqrt(results['mean_squared_error'])
            results['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
            
            # Residuos
            residuals = y_test - y_pred
            results['residuals'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals))
            }
        
        # Importancia de características si está disponible
        if hasattr(model, "feature_importances_"):
            results['feature_importances'] = model.feature_importances_.tolist()
        
        # Si se proporcionaron datos de entrenamiento, comparar rendimiento
        if X_train is not None and y_train is not None:
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results['train_score'] = train_score
            results['test_score'] = test_score
            results['overfitting_diff'] = train_score - test_score
        
        # Añadir información sobre el modelo
        results['model_info'] = {
            'type': type(model).__name__,
            'params': model.get_params(),
        }
        
        return results

    def plot_learning_curve(self, estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
                           return_data=False):
        """
        Genera y visualiza la curva de aprendizaje para un estimador
        
        Args:
            estimator: Estimador scikit-learn (debe implementar fit y predict)
            X: Datos de características
            y: Datos de etiquetas/target
            cv: Estrategia de validación cruzada (int o objeto de validación cruzada)
            train_sizes: Tamaños relativos del conjunto de entrenamiento para evaluar
            return_data: Si True, devuelve también los datos de la curva
            
        Returns:
            BytesIO o dict: Buffer de imagen con la figura o diccionario con datos + figura
        """
        # Calcular curvas de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
            return_times=False
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Crear figura
        plt.figure(figsize=(10, 6))
        plt.title("Curva de Aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Puntuación")
        plt.grid()
        
        # Graficar puntuaciones de entrenamiento y prueba
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r",
                label="Puntuación de entrenamiento")
        plt.plot(train_sizes, test_mean, 'o-', color="g",
                label="Puntuación de validación cruzada")
        
        plt.legend(loc="best")
        
        # Añadir línea de referencia para el rendimiento máximo
        plt.axhline(y=1.0, color='b', linestyle='--', alpha=0.5)
        
        # Calcular brecha de aprendizaje (diferencia entre train y test)
        gap = train_mean[-1] - test_mean[-1]
        plt.annotate(f'Brecha: {gap:.4f}', 
                    xy=(train_sizes[-1], (train_mean[-1] + test_mean[-1])/2),
                    xytext=(train_sizes[-1]*0.8, (train_mean[-1] + test_mean[-1])/2),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        plt.tight_layout()
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        if return_data:
            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores': {
                    'mean': train_mean.tolist(),
                    'std': train_std.tolist()
                },
                'test_scores': {
                    'mean': test_mean.tolist(),
                    'std': test_std.tolist()
                },
                'gap': float(gap),
                'plot': buf
            }
        
        return buf

    def plot_confusion_matrix(self, model, X_test, y_test, class_names=None, normalize=True,
                             cmap=plt.cm.Blues, return_data=False):
        """
        Genera y visualiza la matriz de confusión para un modelo de clasificación
        
        Args:
            model: Modelo de clasificación entrenado (debe implementar predict)
            X_test: Datos de características de prueba
            y_test: Datos de etiquetas/target de prueba
            class_names: Nombres de las clases (si no se especifica, usa índices)
            normalize: Si True, normaliza los valores por filas
            cmap: Esquema de colores matplotlib
            return_data: Si True, devuelve también los datos de la matriz
            
        Returns:
            BytesIO o dict: Buffer de imagen con la figura o diccionario con datos + figura
        """
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Si no hay nombres de clases proporcionados, usar índices
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
            
        # Normalizar si se requiere
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_normalized
            title_suffix = " (normalizada)"
            fmt = '.2f'
        else:
            cm_display = cm
            title_suffix = ""
            fmt = 'd'
            
        # Crear figura
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_display, interpolation='nearest', cmap=cmap)
        plt.title(f"Matriz de Confusión{title_suffix}")
        plt.colorbar()
        
        # Configurar marcadores de ejes
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Añadir valores
        thresh = cm_display.max() / 2.
        for i, j in np.ndindex(cm_display.shape):
            plt.text(j, i, format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")
        
        # Añadir etiquetas y leyenda
        plt.tight_layout()
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        if return_data:
            # Calcular métricas adicionales derivadas de la matriz
            diag = np.diag(cm)
            precision = diag / np.sum(cm, axis=0)
            recall = diag / np.sum(cm, axis=1)
            
            return {
                'confusion_matrix': cm.tolist(),
                'confusion_matrix_normalized': cm_normalized.tolist() if normalize else None,
                'class_names': class_names,
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'plot': buf
            }
        
        return buf

# Prueba del módulo si se ejecuta directamente
if __name__ == '__main__':
    print("Módulo de evaluación avanzada de modelos de Machine Learning")
    
    evaluator = ModelEvaluator()
    print("\nClase inicializada exitosamente")
