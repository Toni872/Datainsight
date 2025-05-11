#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para visualizaciones avanzadas con sistema de caché
Este módulo proporciona funciones para crear visualizaciones y almacenarlas en caché
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap

# Importar gestor de caché
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_manager import CacheManager

# Configurar estilo visual global
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")

# Crear instancia global de gestor de caché
cache = CacheManager()

class AdvancedVisualizer:
    """Clase para crear visualizaciones avanzadas con sistema de caché"""
    
    def __init__(self, use_cache=True):
        """
        Inicializa el visualizador
        
        Args:
            use_cache (bool): Si se debe usar el sistema de caché
        """
        self.use_cache = use_cache
        self.cache_manager = cache
        
        # Configurar paletas de colores personalizadas
        self.color_palettes = {
            'default': sns.color_palette("muted"),
            'divergent': sns.color_palette("RdBu_r", 11),
            'sequential': sns.color_palette("viridis", 9),
            'categorical': sns.color_palette("Set2", 8),
            'correlation': LinearSegmentedColormap.from_list("corr", 
                                                           ["#4169E1", "#FFFFFF", "#DC143C"])
        }
    
    def _get_from_cache(self, plot_type, **kwargs):
        """
        Intenta recuperar una visualización de la caché
        
        Args:
            plot_type (str): Tipo de visualización
            **kwargs: Parámetros de la visualización
            
        Returns:
            BytesIO or None: Buffer con la visualización o None si no está en caché
        """
        if not self.use_cache:
            return None
            
        # Preparar parámetros para clave de caché
        params = {'plot_type': plot_type}
        params.update(kwargs)
        
        # Eliminar parámetros no hashables (como arrays NumPy o DataFrames)
        for k, v in list(params.items()):
            if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
                # Reemplazar con hash de su representación
                if isinstance(v, np.ndarray):
                    params[k] = str(hash(v.tobytes()))
                else:
                    params[k] = str(hash(pd.util.hash_pandas_object(v).sum()))
            elif not isinstance(v, (str, int, float, bool, tuple, list, dict)) or v is None:
                # Eliminar parámetros no hashables
                params.pop(k)
        
        # Obtener de caché
        return self.cache_manager.get(params)
    
    def _save_to_cache(self, buf, plot_type, **kwargs):
        """
        Guarda una visualización en la caché
        
        Args:
            buf (BytesIO): Buffer con la visualización
            plot_type (str): Tipo de visualización
            **kwargs: Parámetros de la visualización
            
        Returns:
            str: Clave de caché generada
        """
        if not self.use_cache:
            return None
            
        # Preparar parámetros para clave de caché
        params = {'plot_type': plot_type}
        params.update(kwargs)
        
        # Eliminar parámetros no hashables (como arrays NumPy o DataFrames)
        for k, v in list(params.items()):
            if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
                # Reemplazar con hash de su representación
                if isinstance(v, np.ndarray):
                    params[k] = str(hash(v.tobytes()))
                else:
                    params[k] = str(hash(pd.util.hash_pandas_object(v).sum()))
            elif not isinstance(v, (str, int, float, bool, tuple, list, dict)) or v is None:
                # Eliminar parámetros no hashables
                params.pop(k)
        
        # Resetear posición del buffer y guardar en caché
        buf.seek(0)
        return self.cache_manager.put(params, buf)
    
    def correlation_matrix(self, data, method='pearson', figsize=(10, 8), 
                         title="Matriz de Correlación", vmin=-1, vmax=1):
        """
        Crea una matriz de correlación visualizada como un heatmap
        
        Args:
            data (pd.DataFrame): DataFrame con variables numéricas
            method (str): Método de correlación ('pearson', 'spearman', 'kendall')
            figsize (tuple): Tamaño de la figura
            title (str): Título del gráfico
            vmin (float): Valor mínimo para la escala de colores
            vmax (float): Valor máximo para la escala de colores
            
        Returns:
            BytesIO: Buffer con la imagen generada
        """
        # Verificar en caché
        cache_key = f"correlation_matrix_{method}_{figsize}_{title}_{vmin}_{vmax}"
        cached = self._get_from_cache('correlation_matrix', 
                                   data=data, 
                                   method=method,
                                   figsize=figsize,
                                   title=title)
        if cached:
            return cached
        
        # Calcular matriz de correlación
        corr_matrix = data.corr(method=method)
        
        # Crear figura
        plt.figure(figsize=figsize)
        
        # Crear heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = self.color_palettes['correlation']
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap,
            vmax=vmax, 
            vmin=vmin, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .8},
            annot=True, 
            fmt=".2f"
        )
        
        plt.title(title)
        plt.tight_layout()
        
        # Guardar en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Guardar en caché y devolver
        self._save_to_cache(buf, 'correlation_matrix', 
                         data=data, 
                         method=method,
                         figsize=figsize,
                         title=title)
        
        buf.seek(0)
        return buf
    
    def feature_importance(self, importance_values, feature_names=None, top_n=None,
                         figsize=(10, 6), title="Importancia de Características",
                         color=None, horizontal=True):
        """
        Crea un gráfico de barras para la importancia de características
        
        Args:
            importance_values (array): Array con valores de importancia
            feature_names (list): Lista con nombres de características
            top_n (int): Mostrar solo las top N características
            figsize (tuple): Tamaño de la figura
            title (str): Título del gráfico
            color (str): Color de las barras
            horizontal (bool): Si True, barras horizontales
            
        Returns:
            BytesIO: Buffer con la imagen generada
        """
        # Verificar en caché
        cached = self._get_from_cache('feature_importance', 
                                   importance_values=importance_values, 
                                   feature_names=feature_names,
                                   top_n=top_n,
                                   figsize=figsize,
                                   title=title,
                                   color=color,
                                   horizontal=horizontal)
        if cached:
            return cached
            
        # Preparar datos
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance_values))]
            
        # Crear DataFrame para facilitar el ordenamiento
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Filtrar por top_n si es necesario
        if top_n is not None and top_n < len(importance_df):
            importance_df = importance_df.head(top_n)
        
        # Crear figura
        plt.figure(figsize=figsize)
        
        # Definir color si no se especifica
        if color is None:
            color = self.color_palettes['categorical'][0]
        
        # Crear gráfico de barras
        if horizontal:
            importance_df = importance_df.sort_values('importance')
            sns.barplot(x='importance', y='feature', data=importance_df, color=color)
            plt.xlabel('Importancia')
            plt.ylabel('Característica')
        else:
            sns.barplot(x='feature', y='importance', data=importance_df, color=color)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Característica')
            plt.ylabel('Importancia')
        
        plt.title(title)
        plt.tight_layout()
        
        # Guardar en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Guardar en caché y devolver
        self._save_to_cache(buf, 'feature_importance', 
                         importance_values=importance_values, 
                         feature_names=feature_names,
                         top_n=top_n,
                         figsize=figsize,
                         title=title,
                         color=color,
                         horizontal=horizontal)
        
        buf.seek(0)
        return buf
    
    def distribution_plot(self, data, x=None, hue=None, kde=True, figsize=(10, 6),
                        title=None, color=None, bins=None):
        """
        Crea un gráfico de distribución (histograma con KDE opcional)
        
        Args:
            data (DataFrame or Series): Datos a visualizar
            x (str): Nombre de la columna a visualizar (si data es DataFrame)
            hue (str): Variable para separar por categorías
            kde (bool): Si se debe mostrar la estimación de densidad kernel
            figsize (tuple): Tamaño de la figura
            title (str): Título del gráfico
            color (str): Color del histograma
            bins (int): Número de bins para el histograma
            
        Returns:
            BytesIO: Buffer con la imagen generada
        """
        # Verificar en caché
        cached = self._get_from_cache('distribution_plot', 
                                   data=data, 
                                   x=x,
                                   hue=hue,
                                   kde=kde,
                                   figsize=figsize,
                                   title=title,
                                   color=color,
                                   bins=bins)
        if cached:
            return cached
            
        # Crear figura
        plt.figure(figsize=figsize)
        
        # Crear histograma
        if isinstance(data, pd.DataFrame):
            if x is None:
                raise ValueError("Debe especificar la columna 'x' cuando data es un DataFrame")
            sns.histplot(data=data, x=x, hue=hue, kde=kde, bins=bins, color=color)
        else:
            sns.histplot(data, kde=kde, bins=bins, color=color)
        
        if title:
            plt.title(title)
        plt.tight_layout()
        
        # Guardar en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Guardar en caché y devolver
        self._save_to_cache(buf, 'distribution_plot', 
                         data=data, 
                         x=x,
                         hue=hue,
                         kde=kde,
                         figsize=figsize,
                         title=title,
                         color=color,
                         bins=bins)
        
        buf.seek(0)
        return buf


# Ejemplo de uso
if __name__ == "__main__":
    # Crear visualizador
    visualizer = AdvancedVisualizer()
    
    # Crear datos de ejemplo
    np.random.seed(42)
    data = pd.DataFrame({
        'var1': np.random.normal(0, 1, 100),
        'var2': np.random.normal(1, 2, 100),
        'var3': np.random.normal(-1, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Crear matriz de correlación
    corr_buf = visualizer.correlation_matrix(data.select_dtypes(include='number'))
    
    # Crear importancia de características simulada
    importance = np.array([0.5, 0.3, 0.2])
    features = ['var1', 'var2', 'var3']
    imp_buf = visualizer.feature_importance(importance, features)
    
    # Crear distribución
    dist_buf = visualizer.distribution_plot(data, x='var1', hue='category', kde=True)
    
    print("Visualizaciones creadas y almacenadas en caché.")
    print("Estadísticas de caché:")
    print(visualizer.cache_manager.get_stats())
