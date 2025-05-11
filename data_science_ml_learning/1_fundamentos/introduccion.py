#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Introducción a Ciencia de Datos y Machine Learning
# Este script sirve como una introducción básica a las herramientas principales que usaremos en nuestro proyecto.

# Importar las bibliotecas principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ejemplo básico de NumPy")
print("-----------------------")
# Crear un array de NumPy
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Media: {arr.mean()}")
print(f"Desviación estándar: {arr.std()}")

# Matriz 2D
matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nMatriz:\n{matriz}")
print(f"Forma: {matriz.shape}")

print("\nEjemplo básico de Pandas")
print("-----------------------")
# Crear un DataFrame de Pandas
df = pd.DataFrame({
    'Nombre': ['Ana', 'Juan', 'María', 'Pedro', 'Lucía'],
    'Edad': [25, 30, 22, 40, 35],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Málaga']
})

print("DataFrame de ejemplo:")
print(df)

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nPróximos pasos")
print("--------------")
print("Este script es solo una introducción básica. En los siguientes notebooks y scripts, exploraremos conceptos más avanzados como:")
print("1. Limpieza y preprocesamiento de datos")
print("2. Análisis exploratorio de datos (EDA)")
print("3. Algoritmos de machine learning")
print("4. Evaluación de modelos")
print("5. Deep learning")

# Para ejecutar este script y ver los resultados, simplemente ejecútalo con:
# python introduccion.py