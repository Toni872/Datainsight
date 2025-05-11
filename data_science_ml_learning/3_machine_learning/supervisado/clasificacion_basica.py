#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clasificación básica usando algoritmos de machine learning
Este script demuestra cómo crear un clasificador básico utilizando scikit-learn.
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def crear_datos_prueba():
    """Crea datos de ejemplo para demostración si no se proporciona un dataset"""
    # Crear datos sintéticos para clasificación
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 4))  # 100 muestras, 4 características
    # Clase binaria basada en la suma de características
    y = (X.sum(axis=1) > 0).astype(int)  
    
    # Convertir a DataFrame para mejor manejo
    df = pd.DataFrame(
        X, 
        columns=[f'caracteristica_{i+1}' for i in range(X.shape[1])]
    )
    df['clase'] = y
    
    return df

def cargar_dataset(nombre_dataset):
    """Intenta cargar un dataset desde la carpeta datasets"""
    try:
        ruta = f"../../datasets/{nombre_dataset}.csv"
        return pd.read_csv(ruta)
    except Exception as e:
        print(f"Error al cargar dataset {nombre_dataset}: {str(e)}")
        print("Usando datos de prueba en su lugar.")
        return crear_datos_prueba()

def entrenar_modelo(df, nombre_dataset):
    """Entrena un modelo de clasificación con los datos proporcionados"""
    # Separar características y variable objetivo
    if 'clase' in df.columns:
        y = df['clase']
        X = df.drop('clase', axis=1)
    else:
        # Si no hay columna 'clase', usar última columna como objetivo
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = modelo.predict(X_test_scaled)
    precision = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    # Preparar resultados
    importancia_caracteristicas = dict(
        zip(X.columns, modelo.feature_importances_)
    )
    
    resultado = {
        "dataset": nombre_dataset,
        "num_muestras": len(df),
        "num_caracteristicas": X.shape[1],
        "precision": precision,
        "importancia_caracteristicas": {
            k: round(float(v), 4) 
            for k, v in sorted(
                importancia_caracteristicas.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
    }
    
    return resultado

def main():
    # Obtener nombre del dataset desde argumentos, o usar "demo" por defecto
    nombre_dataset = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if nombre_dataset == "demo":
        df = crear_datos_prueba()
    else:
        df = cargar_dataset(nombre_dataset)
    
    # Entrenar y evaluar modelo
    resultado = entrenar_modelo(df, nombre_dataset)
    
    # Imprimir resultados en formato JSON para fácil integración con la aplicación web
    print(json.dumps(resultado, indent=2))

if __name__ == "__main__":
    main()