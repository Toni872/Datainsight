#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modelos Avanzados de Aprendizaje No Supervisado
Este módulo implementa algoritmos avanzados de clustering y reducción de dimensionalidad
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import (
    PCA, KernelPCA, FastICA, TruncatedSVD, NMF, 
    FactorAnalysis
)
from sklearn.manifold import (
    TSNE, MDS, Isomap, LocallyLinearEmbedding, 
    SpectralEmbedding
)
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, mutual_info_score
)
from sklearn.preprocessing import StandardScaler

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class UnsupervisedModelTrainer:
    """Clase para entrenar modelos avanzados de aprendizaje no supervisado"""
    
    def __init__(self):
        """Inicializa el entrenador de modelos no supervisados"""
        self.models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.models_path, exist_ok=True)
        
        # Mapear nombres de modelos a sus clases
        self.clustering_models = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'agglomerative': AgglomerativeClustering,
            'spectral_clustering': SpectralClustering,
            'gaussian_mixture': GaussianMixture,
            'birch': Birch
        }
        
        self.decomposition_models = {
            'pca': PCA,
            'kernel_pca': KernelPCA,
            'fast_ica': FastICA,
            'truncated_svd': TruncatedSVD,
            'nmf': NMF,
            'factor_analysis': FactorAnalysis
        }
        
        self.manifold_models = {
            'tsne': TSNE,
            'mds': MDS,
            'isomap': Isomap,
            'lle': LocallyLinearEmbedding,
            'spectral_embedding': SpectralEmbedding
        }
        
    def get_available_models(self, category):
        """
        Devuelve los modelos disponibles para la categoría especificada
        
        Args:
            category (str): 'clustering', 'decomposition' o 'manifold'
            
        Returns:
            dict: Diccionario con los modelos disponibles
        """
        if category == 'clustering':
            return self.clustering_models
        elif category == 'decomposition':
            return self.decomposition_models
        elif category == 'manifold':
            return self.manifold_models
        else:
            categories = ['clustering', 'decomposition', 'manifold']
            raise ValueError(f"Categoría {category} no válida. Opciones: {categories}")
    
    def find_optimal_clusters(self, X, method='kmeans', max_clusters=10):
        """
        Encuentra el número óptimo de clusters para un conjunto de datos
        
        Args:
            X: Datos para clusterizar
            method (str): Método de clustering ('kmeans' o 'gaussian_mixture')
            max_clusters (int): Número máximo de clusters a evaluar
            
        Returns:
            dict: Resultados del análisis
        """
        results = {
            'silhouette_scores': [],
            'calinski_scores': [],
            'davies_bouldin_scores': [],
            'n_clusters': list(range(2, max_clusters + 1))
        }
        
        # Evaluar cada número de clusters
        for n_clusters in range(2, max_clusters + 1):
            # Crear y entrenar el modelo
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'gaussian_mixture':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            else:
                raise ValueError(f"Método {method} no soportado para encontrar clusters óptimos")
            
            # Entrenar y obtener etiquetas
            cluster_labels = model.fit_predict(X)
            
            # Calcular métricas
            if len(np.unique(cluster_labels)) > 1:  # Asegurar que hay más de un cluster
                results['silhouette_scores'].append(silhouette_score(X, cluster_labels))
                results['calinski_scores'].append(calinski_harabasz_score(X, cluster_labels))
                results['davies_bouldin_scores'].append(davies_bouldin_score(X, cluster_labels))
            else:
                results['silhouette_scores'].append(0)
                results['calinski_scores'].append(0)
                results['davies_bouldin_scores'].append(float('inf'))
        
        # Encontrar número óptimo de clusters según cada métrica
        results['optimal_n_clusters_silhouette'] = results['n_clusters'][np.argmax(results['silhouette_scores'])]
        results['optimal_n_clusters_calinski'] = results['n_clusters'][np.argmax(results['calinski_scores'])]
        results['optimal_n_clusters_davies'] = results['n_clusters'][np.argmin(results['davies_bouldin_scores'])]
        
        # Sugerir el número óptimo basado en la mayoría de votos
        optimal_votes = [
            results['optimal_n_clusters_silhouette'],
            results['optimal_n_clusters_calinski'],
            results['optimal_n_clusters_davies']
        ]
        results['suggested_n_clusters'] = max(set(optimal_votes), key=optimal_votes.count)
        
        return results
    
    def train_clustering_model(self, model_name, X, n_clusters=None, params=None):
        """
        Entrena un modelo de clustering
        
        Args:
            model_name (str): Nombre del modelo a entrenar
            X: Datos para clusterizar
            n_clusters (int): Número de clusters (para métodos que lo requieren)
            params (dict): Parámetros adicionales para el modelo
            
        Returns:
            dict: Resultados del entrenamiento
        """
        # Verificar si el modelo solicitado está disponible
        if model_name not in self.clustering_models:
            available = list(self.clustering_models.keys())
            raise ValueError(f"Modelo {model_name} no disponible. Opciones: {available}")
            
        # Crear configuración del modelo
        model_params = params if params else {}
        
        # Para modelos que requieren número de clusters
        if model_name in ['kmeans', 'agglomerative', 'spectral_clustering', 'gaussian_mixture', 'birch']:
            if n_clusters is None:
                n_clusters = 3  # Valor predeterminado
            
            if model_name == 'kmeans':
                model_params['n_clusters'] = n_clusters
                model_params['random_state'] = 42
            elif model_name == 'agglomerative':
                model_params['n_clusters'] = n_clusters
            elif model_name == 'spectral_clustering':
                model_params['n_clusters'] = n_clusters
                model_params['random_state'] = 42
            elif model_name == 'gaussian_mixture':
                model_params['n_components'] = n_clusters
                model_params['random_state'] = 42
            elif model_name == 'birch':
                model_params['n_clusters'] = n_clusters
        
        # Crear modelo
        model_class = self.clustering_models[model_name]
        model = model_class(**model_params)
        
        # Entrenar modelo
        cluster_labels = model.fit_predict(X)
        
        # Calcular métricas si hay más de un cluster
        results = {
            'model_name': model_name,
            'params': model_params,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        unique_clusters = np.unique(cluster_labels)
        results['n_clusters_found'] = len(unique_clusters)
        results['clusters'] = unique_clusters.tolist()
        results['cluster_sizes'] = {
            int(c): int(np.sum(cluster_labels == c)) 
            for c in unique_clusters
        }
        
        # Calcular métricas si hay más de un cluster
        if len(unique_clusters) > 1:
            results['silhouette_score'] = float(silhouette_score(X, cluster_labels))
            results['calinski_harabasz_score'] = float(calinski_harabasz_score(X, cluster_labels))
            results['davies_bouldin_score'] = float(davies_bouldin_score(X, cluster_labels))
        
        # Añadir etiquetas de cluster
        results['cluster_labels'] = cluster_labels.tolist()
        
        return results, model
    
    def train_dimensionality_reduction(self, model_name, X, n_components=2, params=None):
        """
        Entrena un modelo de reducción de dimensionalidad
        
        Args:
            model_name (str): Nombre del modelo a entrenar
            X: Datos para reducir dimensiones
            n_components (int): Número de componentes/dimensiones a mantener
            params (dict): Parámetros adicionales para el modelo
            
        Returns:
            dict: Resultados y datos transformados
        """
        # Determinar la categoría del modelo
        category = None
        if model_name in self.decomposition_models:
            model_dict = self.decomposition_models
            category = 'decomposition'
        elif model_name in self.manifold_models:
            model_dict = self.manifold_models
            category = 'manifold'
        else:
            available_decomp = list(self.decomposition_models.keys())
            available_manifold = list(self.manifold_models.keys())
            raise ValueError(
                f"Modelo {model_name} no disponible. "
                f"Opciones de descomposición: {available_decomp}, "
                f"Opciones de manifold: {available_manifold}"
            )
            
        # Crear configuración del modelo
        model_params = params if params else {}
        
        # Añadir número de componentes
        model_params['n_components'] = n_components
        
        # Para modelos estocásticos, añadir semilla aleatoria
        if model_name in ['tsne', 'mds', 'isomap', 'lle', 'spectral_embedding']:
            if 'random_state' not in model_params:
                model_params['random_state'] = 42
                
        # Crear modelo
        model_class = model_dict[model_name]
        model = model_class(**model_params)
        
        # Aplicar reducción de dimensionalidad
        X_transformed = model.fit_transform(X)
        
        # Preparar resultados
        results = {
            'model_name': model_name,
            'category': category,
            'params': model_params,
            'n_samples': X.shape[0],
            'original_n_features': X.shape[1],
            'reduced_n_features': X_transformed.shape[1]
        }
        
        # Añadir varianza explicada si está disponible (para PCA y similares)
        if hasattr(model, 'explained_variance_ratio_'):
            results['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
            results['cumulative_explained_variance'] = np.cumsum(model.explained_variance_ratio_).tolist()
        
        # Añadir componentes/vectores si están disponibles
        if hasattr(model, 'components_'):
            results['components'] = model.components_.tolist()
        
        return results, X_transformed, model
    
    def create_cluster_visualization(self, X, labels, title="Visualización de Clusters"):
        """
        Crea una visualización de clusters en 2D o 3D
        
        Args:
            X: Datos para visualizar (puede ser original o reducido en dimensionalidad)
            labels: Etiquetas de cluster
            title (str): Título para la visualización
            
        Returns:
            buf: Buffer de imagen con la visualización
        """
        from io import BytesIO
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Crear colores distintos para cada cluster
        n_clusters = len(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        cmap = ListedColormap(colors)
        
        # Reducir dimensiones si es necesario
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_vis = pca.fit_transform(X)
            dim_reducer = 'PCA'
        else:
            X_vis = X
            dim_reducer = None
        
        # Crear visualización en 2D o 3D
        if X_vis.shape[1] >= 3:
            # Visualización 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                X_vis[:, 0], X_vis[:, 1], X_vis[:, 2],
                c=labels, cmap=cmap, alpha=0.8, s=50,
                edgecolors='w', linewidth=0.5
            )
            
            ax.set_title(title)
            ax.set_xlabel(f'Componente 1')
            ax.set_ylabel(f'Componente 2')
            ax.set_zlabel(f'Componente 3')
            
            if dim_reducer:
                fig.text(0.5, 0.01, f'Dimensiones reducidas usando {dim_reducer}', 
                         ha='center', fontsize=10)
            
            # Añadir leyenda
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=cmap(i), markersize=10, 
                          label=f'Cluster {i}')
                for i in range(n_clusters)
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
        else:
            # Visualización 2D
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(
                X_vis[:, 0], X_vis[:, 1],
                c=labels, cmap=cmap, alpha=0.8, s=50,
                edgecolors='w', linewidth=0.5
            )
            
            ax.set_title(title)
            ax.set_xlabel(f'Componente 1')
            ax.set_ylabel(f'Componente 2')
            
            if dim_reducer:
                fig.text(0.5, 0.01, f'Dimensiones reducidas usando {dim_reducer}', 
                         ha='center', fontsize=10)
            
            # Añadir leyenda
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=cmap(i), markersize=10, 
                          label=f'Cluster {i}')
                for i in range(n_clusters)
            ]
            ax.legend(handles=legend_elements, loc='best')
        
        # Guardar figura en buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return buf

    def perform_clustering(self, X, algorithm, **kwargs):
        """
        Realiza clustering con el algoritmo especificado
        
        Args:
            X (array): Datos para clustering
            algorithm (str): Nombre del algoritmo de clustering
            **kwargs: Parámetros adicionales para el algoritmo
            
        Returns:
            dict: Resultados del clustering
        """
        if algorithm not in self.clustering_models:
            available = list(self.clustering_models.keys())
            raise ValueError(f"Algoritmo '{algorithm}' no disponible. Opciones: {available}")
        
        # Preprocesar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear el modelo
        if algorithm == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            model = self.clustering_models[algorithm](n_clusters=n_clusters, random_state=42)
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = self.clustering_models[algorithm](eps=eps, min_samples=min_samples)
        elif algorithm == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', 3)
            linkage = kwargs.get('linkage', 'ward')
            model = self.clustering_models[algorithm](n_clusters=n_clusters, linkage=linkage)
        elif algorithm == 'spectral_clustering':
            n_clusters = kwargs.get('n_clusters', 3)
            model = self.clustering_models[algorithm](n_clusters=n_clusters, random_state=42)
        elif algorithm == 'gaussian_mixture':
            n_components = kwargs.get('n_clusters', 3)  # n_clusters o n_components
            model = self.clustering_models[algorithm](n_components=n_components, random_state=42)
        else:
            model = self.clustering_models[algorithm](**kwargs)
        
        # Aplicar clustering
        labels = model.fit_predict(X_scaled)
        
        # Calcular métricas (si son aplicables)
        results = {
            'labels': labels,
            'n_clusters': len(set([l for l in labels if l != -1])),  # Excluir outliers (-1 en DBSCAN)
            'cluster_sizes': dict([(str(l), int((labels == l).sum())) for l in set(labels)])
        }
        
        # Calcular métricas de calidad del clustering (si hay más de 1 cluster)
        if len(set(labels)) > 1 and -1 not in labels:
            try:
                results['silhouette_score'] = silhouette_score(X_scaled, labels)
            except:
                pass
            
            try:
                results['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
            except:
                pass
            
            try:
                results['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
            except:
                pass
        
        # Añadir el modelo y scaler para futuros usos
        results['model'] = model
        results['scaler'] = scaler
        
        # Generar visualización 2D si se solicita
        if kwargs.get('visualize', False):
            results['visualization'] = self._generate_cluster_visualization(X_scaled, labels, algorithm)
        
        return results
        
    def reduce_dimensions(self, X, method='pca', n_components=2, **kwargs):
        """
        Reduce la dimensionalidad de los datos
        
        Args:
            X (array): Datos a reducir
            method (str): Método de reducción de dimensionalidad
            n_components (int): Número de componentes/dimensiones
            **kwargs: Parámetros adicionales para el método
            
        Returns:
            dict: Datos transformados y modelo
        """
        # Verificar método
        decomposition_methods = list(self.decomposition_models.keys())
        manifold_methods = list(self.manifold_models.keys())
        
        if method not in decomposition_methods and method not in manifold_methods:
            raise ValueError(f"Método '{method}' no disponible. Opciones: {decomposition_methods + manifold_methods}")
        
        # Preprocesar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Seleccionar modelo adecuado
        if method in decomposition_methods:
            if method == 'pca':
                model = self.decomposition_models[method](n_components=n_components, random_state=42)
            elif method == 'kernel_pca':
                kernel = kwargs.get('kernel', 'rbf')
                model = self.decomposition_models[method](n_components=n_components, kernel=kernel, random_state=42)
            elif method == 'truncated_svd':
                model = self.decomposition_models[method](n_components=n_components, random_state=42)
            else:
                model = self.decomposition_models[method](n_components=n_components, **kwargs)
        else:  # Manifold learning
            if method == 'tsne':
                perplexity = kwargs.get('perplexity', 30.0)
                model = self.manifold_models[method](n_components=n_components, perplexity=perplexity, random_state=42)
            elif method == 'mds':
                model = self.manifold_models[method](n_components=n_components, random_state=42)
            else:
                model = self.manifold_models[method](n_components=n_components, **kwargs)
        
        # Aplicar reducción de dimensionalidad
        transformed_data = model.fit_transform(X_scaled)
        
        # Calcular la varianza explicada para PCA
        explained_variance_ratio = None
        if method == 'pca':
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
            cumulative_variance = np.cumsum(model.explained_variance_ratio_).tolist()
        
        results = {
            'transformed_data': transformed_data,
            'model': model,
            'scaler': scaler,
            'n_components': n_components,
            'original_shape': X.shape
        }
        
        if explained_variance_ratio:
            results['explained_variance_ratio'] = explained_variance_ratio
            results['cumulative_variance'] = cumulative_variance
        
        return results
        
    def _generate_cluster_visualization(self, X, labels, algorithm):
        """
        Genera una visualización 2D del clustering
        
        Args:
            X (array): Datos escalados
            labels (array): Etiquetas de cluster
            algorithm (str): Nombre del algoritmo usado
            
        Returns:
            str: Representación codificada en base64 de la imagen
        """
        # Reducir a 2D para visualización si es necesario
        if X.shape[1] > 2:
            # Usar PCA para reducción dimensional
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        # Crear gráfico
        plt.figure(figsize=(10, 8))
        
        # Determinar colores para cada cluster
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:  # Ruido (para DBSCAN)
                col = 'black'
            
            class_member_mask = (labels == k)
            xy = X_2d[class_member_mask]
            plt.scatter(
                xy[:, 0], xy[:, 1],
                s=50, c=[col], label=f'Cluster {k}',
                edgecolor='k', alpha=0.7
            )
        
        plt.title(f'Clustering con {algorithm}')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
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
        
    def get_available_algorithms(self):
        """
        Devuelve los algoritmos disponibles
        
        Returns:
            dict: Diccionarios con los algoritmos disponibles por categoría
        """
        return {
            'clustering': list(self.clustering_models.keys()),
            'dimensionality_reduction': list(self.decomposition_models.keys()) + list(self.manifold_models.keys())
        }

# Prueba del módulo si se ejecuta directamente
if __name__ == '__main__':
    print("Módulo de modelos avanzados de Aprendizaje No Supervisado")
    
    # Mostrar modelos disponibles
    trainer = UnsupervisedModelTrainer()
    print("\nModelos de clustering disponibles:")
    for name in trainer.clustering_models.keys():
        print(f"- {name}")
    
    print("\nModelos de descomposición disponibles:")
    for name in trainer.decomposition_models.keys():
        print(f"- {name}")
        
    print("\nModelos de manifold learning disponibles:")
    for name in trainer.manifold_models.keys():
        print(f"- {name}")
