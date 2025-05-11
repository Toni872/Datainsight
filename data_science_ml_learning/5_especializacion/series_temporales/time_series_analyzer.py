#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para análisis y predicción de series temporales
Este módulo implementa funciones para trabajar con datos de series temporales
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Verificar si tenemos dependencias opcionales
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("pmdarima no está instalado. Para usar auto_arima, instale: pip install pmdarima")

try:
    import prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet no está instalado. Para usar Prophet, instale: pip install prophet")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow no está instalado. Para usar LSTM, instale: pip install tensorflow")

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TimeSeriesAnalyzer:
    """Clase para análisis y predicción de series temporales"""
    
    def __init__(self):
        """Inicializa el analizador de series temporales"""
        self.models = {
            'arima': self._fit_arima,
            'sarima': self._fit_sarima
        }
        
        # Añadir modelos opcionales si están disponibles
        if PMDARIMA_AVAILABLE:
            self.models['auto_arima'] = self._fit_auto_arima
            
        if PROPHET_AVAILABLE:
            self.models['prophet'] = self._fit_prophet
            
        if TENSORFLOW_AVAILABLE:
            self.models['lstm'] = self._fit_lstm
    
    def get_available_models(self):
        """Retorna los modelos disponibles para series temporales"""
        return list(self.models.keys())
    
    def check_stationarity(self, series):
        """
        Comprueba si una serie temporal es estacionaria
        
        Args:
            series (pd.Series): Serie temporal a analizar
            
        Returns:
            dict: Resultados del test ADF (Augmented Dickey-Fuller)
        """
        # Test ADF (Augmented Dickey-Fuller)
        adf_result = adfuller(series.dropna())
        
        # Preparar resultados
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05  # Umbral convencional
        }
    
    def decompose_series(self, series, period=None):
        """
        Descompone una serie temporal en tendencia, estacionalidad y residuo
        
        Args:
            series (pd.Series): Serie temporal a descomponer
            period (int): Período para descomposición estacional (opcional)
            
        Returns:
            dict: Resultados de la descomposición y gráfica
        """
        # Si no se proporciona período, intentar detectarlo
        if period is None:
            # Para datos diarios, asumimos periodo semanal
            if isinstance(series.index, pd.DatetimeIndex):
                if series.index.freq == 'D' or series.index.inferred_freq == 'D':
                    period = 7
                elif series.index.freq == 'M' or series.index.inferred_freq == 'M':
                    period = 12
                elif series.index.freq == 'Q' or series.index.inferred_freq == 'Q':
                    period = 4
                else:
                    # Si no podemos inferir, usar 1/4 de los datos
                    period = len(series) // 4
            else:
                # Si no es DatetimeIndex, usar 1/4 de los datos
                period = len(series) // 4
                
            # Asegurar que el período no sea mayor que la mitad de los datos
            period = min(period, len(series) // 2)
            
            # Asegurar que el período sea al menos 2
            period = max(period, 2)
        
        # Descomponer serie
        decomposition = seasonal_decompose(series, period=period)
        
        # Crear gráfica
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        axes[0].plot(series.index, series.values)
        axes[0].set_title('Serie Original')
        axes[0].grid(True)
        
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_title('Tendencia')
        axes[1].grid(True)
        
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_title('Estacionalidad')
        axes[2].grid(True)
        
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_title('Residuo')
        axes[3].grid(True)
        
        plt.tight_layout()
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Preparar componentes para el resultado
        components = {
            'trend': decomposition.trend.dropna().to_dict(),
            'seasonal': decomposition.seasonal.dropna().to_dict(),
            'resid': decomposition.resid.dropna().to_dict()
        }
        
        # Comprobar estacionariedad del residuo
        resid_stationarity = self.check_stationarity(decomposition.resid.dropna())
        
        return {
            'period': period,
            'components': components,
            'resid_stationarity': resid_stationarity,
            'plot': buf
        }
    
    def autocorrelation_analysis(self, series, lags=40):
        """
        Realiza análisis de autocorrelación y autocorrelación parcial
        
        Args:
            series (pd.Series): Serie temporal a analizar
            lags (int): Número de retrasos a considerar
            
        Returns:
            dict: Resultados de ACF y PACF, y gráfica
        """
        # Calcular ACF y PACF
        acf_values = acf(series.dropna(), nlags=lags)
        pacf_values = pacf(series.dropna(), nlags=lags)
        
        # Crear gráfica
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        plot_acf(series.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Función de Autocorrelación (ACF)')
        
        plot_pacf(series.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Función de Autocorrelación Parcial (PACF)')
        
        plt.tight_layout()
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return {
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'plot': buf
        }
    
    def train_model(self, series, model_name, train_size=0.8, **model_params):
        """
        Entrena un modelo de series temporales
        
        Args:
            series (pd.Series): Serie temporal a modelar
            model_name (str): Nombre del modelo a utilizar
            train_size (float): Proporción de datos para entrenamiento
            **model_params: Parámetros específicos del modelo
            
        Returns:
            dict: Resultados del entrenamiento y predicción
        """
        # Verificar modelo disponible
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Modelo {model_name} no disponible. Opciones: {available}")
        
        # Dividir en entrenamiento y prueba
        train_size = int(len(series) * train_size)
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]
        
        # Llamar a la función de ajuste del modelo específico
        results = self.models[model_name](train, test, **model_params)
        
        return results
    
    def _fit_arima(self, train, test, order=(1, 1, 1)):
        """
        Ajusta un modelo ARIMA
        
        Args:
            train (pd.Series): Datos de entrenamiento
            test (pd.Series): Datos de prueba
            order (tuple): Orden (p,d,q) para ARIMA
            
        Returns:
            dict: Resultados del modelo
        """
        # Ajustar modelo
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        # Hacer predicciones
        predictions = model_fit.forecast(steps=len(test))
        
        # Si las predicciones son una serie, convertir a índices de test
        if isinstance(predictions, pd.Series):
            predictions.index = test.index
        
        # Calcular métricas
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        
        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos de entrenamiento
        ax.plot(train.index, train.values, label='Entrenamiento')
        
        # Graficar datos de prueba
        ax.plot(test.index, test.values, label='Prueba')
        
        # Graficar predicciones
        ax.plot(test.index, predictions, label='Predicción', alpha=0.7)
        
        ax.set_title(f'ARIMA{order} - RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return {
            'model_name': 'arima',
            'order': order,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'summary': str(model_fit.summary()),
            'plot': buf
        }
    
    def _fit_sarima(self, train, test, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Ajusta un modelo SARIMA
        
        Args:
            train (pd.Series): Datos de entrenamiento
            test (pd.Series): Datos de prueba
            order (tuple): Orden (p,d,q) para ARIMA
            seasonal_order (tuple): Orden estacional (P,D,Q,s) para SARIMA
            
        Returns:
            dict: Resultados del modelo
        """
        # Ajustar modelo
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        
        # Hacer predicciones
        predictions = model_fit.forecast(steps=len(test))
        
        # Calcular métricas
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        
        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos de entrenamiento
        ax.plot(train.index, train.values, label='Entrenamiento')
        
        # Graficar datos de prueba
        ax.plot(test.index, test.values, label='Prueba')
        
        # Graficar predicciones
        ax.plot(test.index, predictions, label='Predicción', alpha=0.7)
        
        ax.set_title(f'SARIMA{order}x{seasonal_order} - RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return {
            'model_name': 'sarima',
            'order': order,
            'seasonal_order': seasonal_order,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'summary': str(model_fit.summary()),
            'plot': buf
        }
    
    def _fit_auto_arima(self, train, test, max_p=5, max_q=5, max_d=2, seasonal=True, **kwargs):
        """
        Ajusta un modelo auto ARIMA usando pmdarima
        
        Args:
            train (pd.Series): Datos de entrenamiento
            test (pd.Series): Datos de prueba
            max_p, max_q, max_d (int): Valores máximos para p, q, d
            seasonal (bool): Si se debe considerar estacionalidad
            
        Returns:
            dict: Resultados del modelo
        """
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima no está instalado. Instale con: pip install pmdarima")
        
        # Ajustar modelo
        model = pm.auto_arima(
            train,
            start_p=0, max_p=max_p,
            start_q=0, max_q=max_q,
            max_d=max_d,
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            **kwargs
        )
        
        # Hacer predicciones
        predictions, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
        
        # Calcular métricas
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        
        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos de entrenamiento
        ax.plot(train.index, train.values, label='Entrenamiento')
        
        # Graficar datos de prueba
        ax.plot(test.index, test.values, label='Prueba')
        
        # Graficar predicciones
        ax.plot(test.index, predictions, label='Predicción', alpha=0.7)
        
        # Graficar intervalos de confianza
        ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
        
        ax.set_title(f'Auto ARIMA {model.order}x{model.seasonal_order} - RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return {
            'model_name': 'auto_arima',
            'order': model.order,
            'seasonal_order': model.seasonal_order,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions.tolist(),
            'conf_int': conf_int.tolist(),
            'summary': str(model.summary()),
            'plot': buf
        }
    
    def _fit_prophet(self, train, test, **kwargs):
        """
        Ajusta un modelo Prophet
        
        Args:
            train (pd.Series): Datos de entrenamiento
            test (pd.Series): Datos de prueba
            **kwargs: Parámetros para Prophet
            
        Returns:
            dict: Resultados del modelo
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet no está instalado. Instale con: pip install prophet")
        
        # Preparar datos para Prophet (requiere columnas 'ds' y 'y')
        train_prophet = pd.DataFrame({
            'ds': train.index,
            'y': train.values
        })
        
        # Ajustar modelo
        model = Prophet(**kwargs)
        model.fit(train_prophet)
        
        # Preparar dataframe de fechas futuras
        future = pd.DataFrame({'ds': test.index})
        
        # Hacer predicciones
        forecast = model.predict(future)
        
        # Calcular métricas
        mse = mean_squared_error(test.values, forecast['yhat'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test.values, forecast['yhat'])
        
        # Graficar resultados usando Prophet
        fig = model.plot(forecast)
        
        # Añadir datos reales de test
        ax = fig.gca()
        ax.plot(test.index, test.values, 'r.', alpha=0.5, label='Test Data')
        ax.legend()
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Segunda gráfica: componentes
        fig_comp = model.plot_components(forecast)
        
        # Guardar figura de componentes en buffer
        buf_comp = BytesIO()
        plt.tight_layout()
        plt.savefig(buf_comp, format='png', dpi=100)
        plt.close(fig_comp)
        buf_comp.seek(0)
        
        return {
            'model_name': 'prophet',
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': forecast['yhat'].tolist(),
            'lower_bound': forecast['yhat_lower'].tolist(),
            'upper_bound': forecast['yhat_upper'].tolist(),
            'forecast_components': {
                'trend': forecast['trend'].tolist(),
                'weekly': forecast['weekly'].tolist() if 'weekly' in forecast else None,
                'yearly': forecast['yearly'].tolist() if 'yearly' in forecast else None
            },
            'model_params': kwargs,
            'plot': buf,
            'plot_components': buf_comp
        }
    
    def _fit_lstm(self, train, test, n_lag=10, n_neurons=50, epochs=50, batch_size=32):
        """
        Ajusta un modelo LSTM para series temporales
        
        Args:
            train (pd.Series): Datos de entrenamiento
            test (pd.Series): Datos de prueba
            n_lag (int): Número de pasos anteriores a usar como entrada
            n_neurons (int): Número de neuronas en la capa LSTM
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño del batch para entrenamiento
            
        Returns:
            dict: Resultados del modelo
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está instalado. Instale con: pip install tensorflow")
        
        # Escalar datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        test_scaled = scaler.transform(test.values.reshape(-1, 1))
        
        # Crear secuencias para LSTM
        def create_sequences(data, n_lag):
            X, y = [], []
            for i in range(len(data) - n_lag):
                X.append(data[i:i+n_lag])
                y.append(data[i+n_lag])
            return np.array(X), np.array(y)
        
        # Preparar datos de entrenamiento
        X_train, y_train = create_sequences(train_scaled, n_lag)
        
        # Crear modelo LSTM
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(n_lag, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        
        # Entrenar modelo
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Predecir usando el modelo entrenado
        # Para predecir el primer punto de test, usamos los últimos n_lag puntos de train
        predictions = []
        current_batch = train_scaled[-n_lag:].reshape(1, n_lag, 1)
        
        for i in range(len(test)):
            # Predecir el siguiente punto
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred[0])
            
            # Actualizar batch para la siguiente predicción incluyendo el valor predicho
            current_batch = np.append(current_batch[:, 1:, :], 
                                     [[current_pred]], 
                                     axis=1)
        
        # Invertir la escala de las predicciones
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Calcular métricas
        mse = mean_squared_error(test.values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test.values, predictions)
        
        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos de entrenamiento
        ax.plot(train.index, train.values, label='Entrenamiento')
        
        # Graficar datos de prueba
        ax.plot(test.index, test.values, label='Prueba')
        
        # Graficar predicciones
        ax.plot(test.index, predictions, label='Predicción LSTM', alpha=0.7)
        
        ax.set_title(f'LSTM (neurons={n_neurons}, lag={n_lag}) - RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Graficar historia de pérdida
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(history.history['loss'], label='Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Validación')
        ax2.set_title('Pérdida del Modelo LSTM')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Error (MSE)')
        ax2.legend()
        ax2.grid(True)
        
        # Guardar figura en buffer
        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=100)
        plt.close(fig2)
        buf2.seek(0)
        
        return {
            'model_name': 'lstm',
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions.flatten().tolist(),
            'model_params': {
                'n_lag': n_lag,
                'n_neurons': n_neurons,
                'epochs': epochs,
                'batch_size': batch_size
            },
            'loss_history': {
                'train': history.history['loss'],
                'val': history.history['val_loss']
            },
            'plot': buf,
            'loss_plot': buf2
        }

    def forecast_future(self, series, model_name, steps=30, **model_params):
        """
        Realiza una predicción hacia el futuro
        
        Args:
            series (pd.Series): Serie temporal completa
            model_name (str): Nombre del modelo a utilizar
            steps (int): Número de pasos futuros a predecir
            **model_params: Parámetros específicos del modelo
            
        Returns:
            dict: Resultados de la predicción futura
        """
        # Implementación específica para cada modelo
        # En este método, entrenaríamos el modelo con toda la serie
        # y luego extenderíamos la predicción 'steps' hacia el futuro
        
        # Esta es una implementación básica que se puede expandir
        if model_name == 'arima':
            # Entrenar ARIMA con toda la serie
            order = model_params.get('order', (1, 1, 1))
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            
            # Predecir pasos futuros
            forecast = model_fit.forecast(steps=steps)
            
            # Crear fechas futuras para el índice de la predicción
            if isinstance(series.index, pd.DatetimeIndex):
                # Si el índice es de tipo fecha, extender las fechas
                freq = series.index.freq
                if freq is None:
                    freq = pd.infer_freq(series.index)
                
                last_date = series.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)
                forecast = pd.Series(forecast, index=future_dates)
            
            # Graficar resultados
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Graficar datos históricos
            ax.plot(series.index, series.values, label='Histórico')
            
            # Graficar predicción
            ax.plot(forecast.index, forecast.values, label='Predicción', color='red', alpha=0.7)
            
            ax.set_title(f'Predicción Futura ARIMA{order} - {steps} pasos')
            ax.legend()
            ax.grid(True)
            
            # Guardar figura en buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            
            return {
                'model_name': 'arima',
                'forecast': forecast.tolist(),
                'forecast_dates': [str(d) for d in forecast.index],
                'model_params': {'order': order},
                'plot': buf
            }
            
        elif model_name == 'prophet':
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet no está instalado. Instale con: pip install prophet")
                
            # Preparar datos para Prophet
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # Entrenar modelo
            model = Prophet(**model_params)
            model.fit(df)
            
            # Crear dataframe para predicción futura
            if isinstance(series.index, pd.DatetimeIndex):
                # Si el índice es de tipo fecha, extender con date_range
                future = model.make_future_dataframe(periods=steps, freq=series.index.freq)
            else:
                # Si no es fecha, crear una secuencia numérica
                last_idx = series.index[-1]
                future_idx = range(last_idx + 1, last_idx + steps + 1)
                future = pd.DataFrame({'ds': future_idx})
            
            # Hacer predicción
            forecast = model.predict(future)
            
            # Graficar resultados
            fig = model.plot(forecast)
            
            # Guardar figura en buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            
            # Gráfica de componentes
            fig_comp = model.plot_components(forecast)
            
            # Guardar figura de componentes en buffer
            buf_comp = BytesIO()
            plt.tight_layout()
            plt.savefig(buf_comp, format='png', dpi=100)
            plt.close(fig_comp)
            buf_comp.seek(0)
            
            # Extraer solo los nuevos puntos para retornar
            new_points = forecast.iloc[-steps:]
            
            return {
                'model_name': 'prophet',
                'forecast': new_points['yhat'].tolist(),
                'lower_bound': new_points['yhat_lower'].tolist(),
                'upper_bound': new_points['yhat_upper'].tolist(),
                'forecast_dates': [str(d) for d in new_points['ds']],
                'model_params': model_params,
                'plot': buf,
                'plot_components': buf_comp
            }
            
        else:
            raise NotImplementedError(f"Predicción futura no implementada para el modelo {model_name}")

    def decompose_time_series(self, series, model='additive', period=None):
        """
        Descompone una serie temporal en sus componentes
        
        Args:
            series (pd.Series): Serie temporal con índice datetime
            model (str): 'additive' o 'multiplicative'
            period (int): Período estacional (si es None, se detecta automáticamente)
            
        Returns:
            dict: Componentes de la serie temporal (tendencia, estacionalidad, residual)
        """
        import statsmodels.api as sm
        
        # Verificar que la serie tiene índice datetime
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("La serie debe tener un índice de tipo datetime")
        
        # Detectar período si no se proporciona
        if period is None:
            # Detectar frecuencia común
            if series.index.freqstr:
                freq = series.index.freqstr
                if 'D' in freq:  # Diario
                    period = 7  # Semanal
                elif 'M' in freq:  # Mensual
                    period = 12  # Anual
                elif 'H' in freq:  # Horario
                    period = 24  # Diario
                elif 'min' in freq.lower():  # Minutos
                    period = 60  # Horario
                else:
                    period = 12  # Por defecto
            else:
                # Intentar inferir frecuencia
                try:
                    # Calcular la mediana de las diferencias de tiempo
                    deltas = series.index[1:] - series.index[:-1]
                    median_delta = pd.Timedelta(deltas.median())
                    
                    if median_delta < pd.Timedelta('1 hour'):
                        period = 60  # Minutos en una hora
                    elif median_delta < pd.Timedelta('1 day'):
                        period = 24  # Horas en un día
                    elif median_delta < pd.Timedelta('8 days'):
                        period = 7  # Días en una semana
                    elif median_delta < pd.Timedelta('40 days'):
                        period = 30  # Días en un mes
                    else:
                        period = 12  # Meses en un año
                except:
                    period = 12  # Por defecto
        
        # Asegurar que hay suficientes datos para el período
        if len(series) < period * 2:
            raise ValueError(f"La serie es demasiado corta para el período {period}. Se necesitan al menos {period * 2} observaciones.")
        
        # Realizar la descomposición
        decomposition = sm.tsa.seasonal_decompose(
            series, 
            model=model, 
            period=period
        )
        
        # Extraer componentes
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Crear diccionario de resultados
        result = {
            'trend': trend.dropna(),
            'seasonal': seasonal.dropna(),
            'residual': residual.dropna(),
            'period': period,
            'model': model
        }
        
        # Generar gráfica si se solicita
        result['visualization'] = self._visualize_decomposition(decomposition)
        
        return result
        
    def forecast_arima(self, series, periods=10, order=None, seasonal_order=None, return_conf_int=True):
        """
        Realiza pronóstico con modelo ARIMA
        
        Args:
            series (pd.Series): Serie temporal con índice datetime
            periods (int): Número de períodos a pronosticar
            order (tuple): Orden del modelo ARIMA (p,d,q)
            seasonal_order (tuple): Orden estacional (P,D,Q,s)
            return_conf_int (bool): Si se devuelven intervalos de confianza
            
        Returns:
            dict: Pronóstico y métricas
        """
        import statsmodels.api as sm
        
        # Verificar que la serie tiene índice datetime
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("La serie debe tener un índice de tipo datetime")
        
        # Detectar si la serie es estacionaria
        stationarity_results = self.check_stationarity(series)
        is_stationary = stationarity_results['is_stationary']
        
        # Determinar el orden del modelo si no se proporciona
        if order is None:
            # Estimar d basado en estacionariedad
            d = 0 if is_stationary else 1
            # Valores predeterminados conservadores para p y q
            p, q = 1, 1
            order = (p, d, q)
        
        # Crear y ajustar modelo ARIMA/SARIMA
        if seasonal_order:
            model = sm.tsa.SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_type = 'SARIMA'
        else:
            model = sm.tsa.ARIMA(series, order=order)
            model_type = 'ARIMA'
        
        model_fit = model.fit()
        
        # Generar pronóstico
        if return_conf_int:
            forecast = model_fit.get_forecast(steps=periods, alpha=0.05)
            forecast_index = self._generate_forecast_index(series, periods)
            
            # Extraer predicciones e intervalos de confianza
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            if forecast_index is not None:
                mean_forecast.index = forecast_index
                conf_int.index = forecast_index
            
            # Crear dataframe con el pronóstico y los intervalos
            forecast_df = pd.DataFrame({
                'forecast': mean_forecast,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1]
            })
        else:
            forecast_result = model_fit.forecast(steps=periods)
            forecast_index = self._generate_forecast_index(series, periods)
            
            if forecast_index is not None:
                forecast_result.index = forecast_index
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_result
            })
        
        # Calcular métricas de precisión en los datos de entrenamiento
        train_metrics = {}
        try:
            predictions = model_fit.fittedvalues
            train_metrics['mse'] = ((series - predictions) ** 2).mean()
            train_metrics['rmse'] = np.sqrt(train_metrics['mse'])
            train_metrics['mae'] = (series - predictions).abs().mean()
        except:
            pass
        
        # Generar visualización
        visualization = self._visualize_forecast(series, forecast_df)
        
        # Resultados
        result = {
            'model': model_fit,
            'model_type': model_type,
            'order': order,
            'seasonal_order': seasonal_order,
            'forecast_data': forecast_df.reset_index().rename(columns={'index': 'date'}).to_dict('records'),
            'metrics': train_metrics,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'visualization': visualization
        }
        
        return result
        
    def _generate_forecast_index(self, series, periods):
        """
        Genera un índice para los datos de pronóstico
        
        Args:
            series (pd.Series): Serie temporal original
            periods (int): Número de períodos a pronosticar
            
        Returns:
            pd.DatetimeIndex: Índice para el pronóstico
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            return None
            
        # Detectar la frecuencia
        if series.index.freqstr:
            freq = series.index.freqstr
        else:
            # Calcular la mediana de las diferencias de tiempo
            deltas = series.index[1:] - series.index[:-1]
            median_delta = pd.Timedelta(deltas.median())
            
            # Convertir a frecuencia
            if median_delta < pd.Timedelta('1 minute'):
                freq = 'S'  # Segundos
            elif median_delta < pd.Timedelta('1 hour'):
                freq = 'min'  # Minutos
            elif median_delta < pd.Timedelta('1 day'):
                freq = 'H'  # Horas
            elif median_delta < pd.Timedelta('8 days'):
                freq = 'D'  # Días
            elif median_delta < pd.Timedelta('40 days'):
                freq = 'W'  # Semanas
            elif median_delta < pd.Timedelta('370 days'):
                freq = 'M'  # Meses
            else:
                freq = 'Y'  # Años
        
        # Generar índice futuro
        last_date = series.index[-1]
        future_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        
        return future_index
        
    def _visualize_decomposition(self, decomposition):
        """
        Genera una visualización de la descomposición de series temporales
        
        Args:
            decomposition: Objeto de descomposición de statsmodels
            
        Returns:
            str: Representación codificada en base64 de la imagen
        """
        # Crear figura
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Graficar los componentes
        decomposition.observed.plot(ax=axes[0], title='Serie Original')
        decomposition.trend.plot(ax=axes[1], title='Tendencia')
        decomposition.seasonal.plot(ax=axes[2], title='Estacionalidad')
        decomposition.resid.plot(ax=axes[3], title='Residuales')
        
        plt.tight_layout()
        
        # Convertir gráfico a imagen base64
        from io import BytesIO
        import base64
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str
        
    def _visualize_forecast(self, original, forecast_df):
        """
        Genera una visualización del pronóstico
        
        Args:
            original (pd.Series): Serie temporal original
            forecast_df (pd.DataFrame): DataFrame con el pronóstico
            
        Returns:
            str: Representación codificada en base64 de la imagen
        """
        # Crear figura
        plt.figure(figsize=(12, 6))
        
        # Graficar serie original
        plt.plot(original.index, original, label='Observado', color='blue')
        
        # Graficar pronóstico
        plt.plot(forecast_df.index, forecast_df['forecast'], label='Pronóstico', color='red')
        
        # Si hay intervalos de confianza, graficarlos
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            plt.fill_between(
                forecast_df.index,
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                color='pink', alpha=0.3,
                label='Intervalo 95%'
            )
        
        plt.title('Pronóstico de Serie Temporal')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convertir gráfico a imagen base64
        from io import BytesIO
        import base64
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str
        
    def get_available_methods(self):
        """
        Devuelve los métodos disponibles para análisis de series temporales
        
        Returns:
            dict: Métodos disponibles por categoría
        """
        forecasting_methods = ['arima', 'sarima']
        
        if self.prophet_available:
            forecasting_methods.append('prophet')
        
        if self.pmdarima_available:
            forecasting_methods.append('auto_arima')
        
        if self.tensorflow_available:
            forecasting_methods.append('lstm')
            
        return {
            'analysis': ['decomposition', 'autocorrelation', 'stationarity'],
            'forecasting': forecasting_methods
        }

    def forecast_arima(self, series, periods=10, order=(1, 1, 1), alpha=0.05):
        """
        Realiza un pronóstico ARIMA para una serie temporal
        
        Args:
            series (pd.Series): Serie temporal a pronosticar
            periods (int): Número de periodos futuros a pronosticar
            order (tuple): Orden (p,d,q) para ARIMA
            alpha (float): Nivel de confianza para los intervalos de predicción
            
        Returns:
            dict: Resultados del pronóstico
        """
        # Ajustar modelo ARIMA
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        
        # Realizar pronóstico
        forecast_result = model_fit.forecast(steps=periods)
        
        # Si el pronóstico es una serie, extraer los valores
        forecast_values = forecast_result if isinstance(forecast_result, np.ndarray) else forecast_result.values
        
        # Crear índice para el pronóstico (continuando desde el último valor de la serie)
        if isinstance(series.index, pd.DatetimeIndex):
            # Si es un índice de fecha/hora, extender con la misma frecuencia
            freq = pd.infer_freq(series.index)
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1), 
                periods=periods, 
                freq=freq
            )
        else:
            # Si no es un índice de fecha/hora, usar números enteros
            last_idx = series.index[-1]
            forecast_index = range(last_idx + 1, last_idx + periods + 1)
        
        # Crear serie para el pronóstico
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        # Graficar resultados
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos históricos
        ax.plot(series.index, series.values, label='Histórico')
        
        # Graficar pronóstico
        ax.plot(forecast_series.index, forecast_series.values, 'r--', label='Pronóstico')
        
        ax.set_title(f'Pronóstico ARIMA{order} - {periods} periodos')
        ax.legend()
        ax.grid(True)
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return {
            'model_name': 'arima',
            'order': order,
            'forecast': forecast_values.tolist(),
            'forecast_index': [str(idx) for idx in forecast_index],
            'plot': buf
        }

# Prueba del módulo si se ejecuta directamente
if __name__ == '__main__':
    print("Módulo de análisis y predicción de series temporales")
    
    analyzer = TimeSeriesAnalyzer()
    print("\nModelos disponibles:")
    for name in analyzer.get_available_models():
        print(f"- {name}")
