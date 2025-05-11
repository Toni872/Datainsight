# Documentación de Algoritmos Avanzados de Machine Learning

## Algoritmos Implementados

### Modelos Supervisados Avanzados

#### Stacking Classifier
El algoritmo de "stacking" (apilamiento) es una técnica de ensamblado que combina múltiples modelos de clasificación a través de un meta-clasificador. Los modelos base se entrenan en el conjunto de datos completo, luego sus predicciones se utilizan como entrada para entrenar un meta-modelo que aprende a combinarlas óptimamente.

**Arquitectura Implementada:**
* **Estimadores Base**:
  * RandomForest
  * GradientBoosting
  * SVM (con habilitación de probabilidades)
* **Meta-clasificador**: LogisticRegression
* **Validación Cruzada**: 5-fold cross-validation para generar las predicciones de los modelos base

**Ventajas**:
* Mejora la precisión al combinar las fortalezas de diferentes algoritmos
* Reduce el sesgo y la varianza
* Mayor robustez ante diferentes tipos de datos

**Uso recomendado**: Problemas de clasificación complejos donde un solo modelo no proporciona suficiente precisión.

#### Stacking Regressor
Similar al Stacking Classifier pero para problemas de regresión. Combina múltiples modelos de regresión y utiliza un meta-modelo para aprender la mejor combinación de sus predicciones.

**Arquitectura Implementada:**
* **Estimadores Base**:
  * RandomForest Regressor
  * GradientBoosting Regressor
  * Support Vector Regression
* **Meta-regressor**: Linear Regression
* **Validación Cruzada**: 5-fold cross-validation

**Uso recomendado**: Problemas de regresión complejos, especialmente cuando diferentes modelos capturan diferentes aspectos de la relación entre variables.

### Clustering Avanzado

Se han implementado y optimizado varios algoritmos de clustering:

#### K-Means con inicialización avanzada
Implementación mejorada de K-means que utiliza múltiples inicializaciones y selecciona el mejor resultado.

**Características**:
* Múltiples inicializaciones con k-means++
* Evaluación automática con silhouette score
* Visualización de clusters con reducción de dimensionalidad

#### DBSCAN con estimación automática de parámetros
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo que agrupa puntos que están densamente agrupados.

**Características**:
* Estimación automática del parámetro eps basada en vecinos más cercanos
* Detección de ruido (outliers)
* No requiere especificar el número de clusters de antemano

#### Agglomerative Clustering con diferentes métricas
Clustering jerárquico que construye una jerarquía de clusters mediante un enfoque bottom-up.

**Características**:
* Soporte para diferentes métricas de distancia
* Visualización de dendrogramas
* Flexibilidad en el criterio de enlace (ward, complete, average, single)

### Análisis de Series Temporales

#### Descomposición de Series Temporales
Descompone una serie temporal en sus componentes:
* Tendencia
* Estacionalidad
* Residuo (componente aleatorio)

**Aplicaciones**:
* Análisis exploratorio de datos temporales
* Identificación de patrones estacionales
* Detección de anomalías

#### Pronóstico ARIMA
Implementación de modelos ARIMA (AutoRegressive Integrated Moving Average) para pronóstico de series temporales.

**Características**:
* Selección automática de parámetros (p,d,q)
* Intervalos de predicción
* Diagnóstico de residuos

## Técnicas de Evaluación de Modelos

### Evaluación Completa de Modelos
Proporciona una evaluación exhaustiva de modelos con múltiples métricas:

* **Para Clasificación**:
  * Accuracy, Precision, Recall, F1-Score
  * Matriz de Confusión
  * Curva ROC y AUC
  * Curva Precision-Recall
  * Curva de calibración

* **Para Regresión**:
  * R², Error Cuadrático Medio (MSE)
  * Raíz de Error Cuadrático Medio (RMSE)
  * Error Absoluto Medio (MAE)
  * Análisis de residuos

### Curvas de Aprendizaje
Visualización del rendimiento del modelo en función del tamaño del conjunto de entrenamiento para detectar problemas de sesgo/varianza.

**Interpretación**:
* **Alta brecha entre puntuaciones de entrenamiento y validación**: Indica sobreajuste
* **Ambas puntuaciones bajas**: Indica subajuste
* **Curva de validación con potencial de mejora**: Puede beneficiarse de más datos

### Matriz de Confusión Visual
Visualización mejorada de matrices de confusión con valores normalizados para facilitar la interpretación.

## Sistema de Caché para Visualizaciones

Se ha implementado un sistema de caché para optimizar el rendimiento al generar visualizaciones, especialmente útil para la interfaz web.

**Características**:
* Almacenamiento eficiente de visualizaciones
* Expiración automática de caché
* Optimización de rendimiento para dashboards

**Tipos de visualizaciones optimizadas**:
* Matrices de correlación
* Importancia de características
* Distribuciones de variables
* Curvas de aprendizaje
* Matrices de confusión

## Integración con la API

Todos los modelos y visualizaciones están disponibles a través de la API REST implementada en `ml_service_v2.py`.

**Endpoints principales**:
* `/models/available`: Lista modelos disponibles
* `/models/train`: Entrena un modelo con datos proporcionados
* `/clustering/perform`: Realiza clustering en un conjunto de datos
* `/timeseries/analyze`: Analiza una serie temporal
* `/models/evaluate`: Evalúa el rendimiento de un modelo

## Requerimientos del Sistema

Para el funcionamiento completo de todas las características se recomiendan las siguientes librerías:

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
scipy>=1.7.0
statsmodels>=0.13.0
seaborn>=0.11.0
joblib>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
```

Para funcionalidades avanzadas opcionales:
```
pmdarima       # Para modelado ARIMA automático
prophet        # Para modelos de series temporales de Facebook
tensorflow     # Para redes neuronales LSTM
```

## Ejemplos de Uso

Consulte los scripts de prueba en `test_advanced_ml.py` para ejemplos de uso de todas las funcionalidades.
