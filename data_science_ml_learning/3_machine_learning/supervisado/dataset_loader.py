#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Loader - Módulo para cargar y procesar datasets para machine learning
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class DatasetProcessor:
    """Clase para cargar y procesar datasets para machine learning"""
    
    def __init__(self):
        """Inicializa el procesador de datasets"""
        self.datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'datasets')
        self.standard_datasets = {
            'iris': os.path.join(self.datasets_path, 'iris_sample.csv'),
            'wine': None,  # Se cargará desde sklearn
            'breast_cancer': None,  # Se cargará desde sklearn
            'diabetes': None  # Se cargará desde sklearn
        }
        self.encoders = {}
        
    def get_dataset_info(self, dataset_id):
        """
        Obtiene información sobre un dataset
        
        Args:
            dataset_id (str): Identificador del dataset
            
        Returns:
            dict: Información del dataset
        """
        df = self._load_dataset(dataset_id)
        
        if df is None:
            return {
                'error': f'Dataset {dataset_id} no encontrado'
            }
        
        # Determinar el tipo de tarea
        task_type = self._infer_task_type(df)
        
        # Determinar la columna objetivo (última columna por defecto)
        target_col = df.columns[-1]
        
        info = {
            'id': dataset_id,
            'name': dataset_id.replace('_', ' ').title(),
            'rows': len(df),
            'columns': len(df.columns),
            'features': list(df.columns[:-1]),
            'target': target_col,
            'task_type': task_type,
            'preview': df.head(5).to_dict(orient='records'),
            'feature_types': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Si es clasificación, añadir las clases
        if task_type == 'classification':
            info['classes'] = df[target_col].unique().tolist()
            try:
                info['classes'] = sorted(info['classes'])
            except:
                pass  # Si no se pueden ordenar, dejamos como están
        
        return info
    
    def prepare_data(self, dataset_id, test_size=0.2, scaling=True):
        """
        Prepara los datos para entrenamiento y prueba
        
        Args:
            dataset_id (str): Identificador del dataset
            test_size (float): Tamaño del conjunto de prueba (0.0 - 1.0)
            scaling (bool): Si se debe aplicar escalado estándar
            
        Returns:
            dict: Datos preparados para entrenamiento
        """
        df = self._load_dataset(dataset_id)
        
        if df is None:
            return {'error': f'Dataset {dataset_id} no encontrado'}
        
        # Determinar columna objetivo (última columna por defecto)
        target_col = df.columns[-1]
        
        # Separar características y objetivo
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Codificar variable objetivo si es categórica
        if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.encoders[dataset_id] = le
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Aplicar escalado si se solicita
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Convertir a listas para serialización JSON
        data = {
            'X_train': X_train.tolist() if isinstance(X_train, np.ndarray) else X_train,
            'X_test': X_test.tolist() if isinstance(X_test, np.ndarray) else X_test,
            'y_train': y_train.tolist() if isinstance(y_train, np.ndarray) else y_train,
            'y_test': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test,
            'feature_names': list(X.columns),
            'dataset_id': dataset_id
        }
        
        return data
    
    def save_custom_dataset(self, file_path, name, target_column=None, description=None):
        """
        Guarda un dataset personalizado en el directorio de datasets
        
        Args:
            file_path (str): Ruta al archivo CSV
            name (str): Nombre del dataset
            target_column (str, optional): Nombre de la columna objetivo
            description (str, optional): Descripción del dataset
            
        Returns:
            dict: Información del dataset guardado
        """
        try:
            # Cargar el dataset desde el archivo CSV
            df = pd.read_csv(file_path)
            
            # Si no se especifica columna objetivo, usar la última
            if not target_column:
                target_column = df.columns[-1]
            
            # Si la columna objetivo no está en las últimas posiciones,
            # reorganizar para que sea la última (convención)
            if target_column in df.columns and list(df.columns).index(target_column) < len(df.columns) - 1:
                cols = [col for col in df.columns if col != target_column] + [target_column]
                df = df[cols]
            
            # Crear ID para el dataset (slug)
            dataset_id = name.lower().replace(' ', '_')
            
            # Guardar el dataset en el directorio de datasets
            output_path = os.path.join(self.datasets_path, f'{dataset_id}.csv')
            df.to_csv(output_path, index=False)
            
            # Añadir a la lista de datasets estándar
            self.standard_datasets[dataset_id] = output_path
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'name': name,
                'path': output_path,
                'rows': len(df),
                'columns': len(df.columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_datasets(self):
        """
        Lista todos los datasets disponibles
        
        Returns:
            list: Lista de datasets disponibles
        """
        datasets = []
        
        # Datasets predefinidos
        for dataset_id in self.standard_datasets:
            name = dataset_id.replace('_', ' ').title()
            datasets.append({
                'id': dataset_id,
                'name': name,
                'type': 'standard'
            })
        
        # Datasets personalizados (archivos CSV en el directorio de datasets)
        for file_name in os.listdir(self.datasets_path):
            if file_name.endswith('.csv') and not file_name.startswith('_'):
                dataset_id = file_name[:-4]  # Quitar extensión .csv
                if dataset_id not in self.standard_datasets:
                    name = dataset_id.replace('_', ' ').title()
                    datasets.append({
                        'id': dataset_id,
                        'name': name,
                        'type': 'custom',
                        'path': os.path.join(self.datasets_path, file_name)
                    })
        
        return datasets
    
    def _load_dataset(self, dataset_id):
        """
        Carga un dataset por su ID
        
        Args:
            dataset_id (str): Identificador del dataset
            
        Returns:
            pandas.DataFrame: Dataset cargado o None si no se encuentra
        """
        # Verificar datasets estándar
        if dataset_id == 'iris':
            if self.standard_datasets[dataset_id] and os.path.exists(self.standard_datasets[dataset_id]):
                return pd.read_csv(self.standard_datasets[dataset_id])
            else:
                # Si no existe el archivo iris_sample.csv, cargarlo desde sklearn
                from sklearn.datasets import load_iris
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['species'] = [data.target_names[i] for i in data.target]
                return df
        elif dataset_id == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['class'] = data.target
            return df
        elif dataset_id == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif dataset_id == 'diabetes':
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        
        # Verificar datasets personalizados en el directorio
        custom_path = os.path.join(self.datasets_path, f'{dataset_id}.csv')
        if os.path.exists(custom_path):
            return pd.read_csv(custom_path)
        
        # Si se proporciona una ruta completa, intentar cargar directamente
        if os.path.exists(dataset_id):
            return pd.read_csv(dataset_id)
        
        # No se encontró el dataset
        return None
    
    def _infer_task_type(self, df):
        """
        Infiere el tipo de tarea (clasificación o regresión) basado en la variable objetivo
        
        Args:
            df (pandas.DataFrame): Dataset
            
        Returns:
            str: 'classification' o 'regression'
        """
        # La columna objetivo es la última por convención
        target_col = df.columns[-1]
        
        # Si la columna es categórica o tiene pocos valores únicos, es clasificación
        if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
            return 'classification'
        
        # Si la columna es numérica y tiene pocos valores únicos (< 10% de filas),
        # probablemente sea clasificación
        n_unique = len(df[target_col].unique())
        if n_unique < len(df) * 0.1 or n_unique < 10:
            return 'classification'
        
        # De lo contrario, es regresión
        return 'regression'


# Función principal para usar desde línea de comandos
def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Se requiere una acción'}))
        return
    
    action = sys.argv[1]
    processor = DatasetProcessor()
    
    if action == 'get_info':
        if len(sys.argv) < 3:
            print(json.dumps({'error': 'Se requiere un ID de dataset'}))
            return
        dataset_id = sys.argv[2]
        info = processor.get_dataset_info(dataset_id)
        print(json.dumps(info))
    
    elif action == 'prepare_data':
        if len(sys.argv) < 3:
            print(json.dumps({'error': 'Se requiere un ID de dataset'}))
            return
        
        dataset_id = sys.argv[2]
        test_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
        scaling = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
        
        data = processor.prepare_data(dataset_id, test_size, scaling)
        print(json.dumps({'success': True}))
        
        # Guardar datos en un archivo temporal por su tamaño
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'temp_{dataset_id}_data.json')
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        
        print(json.dumps({'success': True, 'temp_file': temp_file}))
    
    elif action == 'list':
        datasets = processor.list_datasets()
        print(json.dumps(datasets))
    
    elif action == 'save':
        if len(sys.argv) < 4:
            print(json.dumps({'error': 'Se requieren ruta y nombre del dataset'}))
            return
        
        file_path = sys.argv[2]
        name = sys.argv[3]
        target_column = sys.argv[4] if len(sys.argv) > 4 else None
        description = sys.argv[5] if len(sys.argv) > 5 else None
        
        result = processor.save_custom_dataset(file_path, name, target_column, description)
        print(json.dumps(result))
    
    else:
        print(json.dumps({'error': 'Acción no reconocida'}))


if __name__ == '__main__':
    main()