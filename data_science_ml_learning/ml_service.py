#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Servicio API para ML usando FastAPI
Este script crea un API REST simple para servir modelos de machine learning
"""

import os
import sys
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Crear la aplicación FastAPI
app = FastAPI(
    title="DataInsight AI - Servicio ML",
    description="API para servir modelos de machine learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, limitar a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos para la API
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: Any
    prediction_proba: Optional[List[float]] = None
    model_name: str

# Almacenamiento en memoria para modelos entrenados
trained_models = {}

# Función para entrenar un modelo iris de ejemplo
def train_iris_model():
    # Cargar el dataset iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un clasificador Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Guardar en el diccionario de modelos
    trained_models["iris_classifier"] = {
        "model": model,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names,
        "accuracy": model.score(X_test, y_test)
    }
    
    print(f"Modelo iris_classifier entrenado con precisión: {model.score(X_test, y_test):.4f}")

# Rutas de la API

@app.get("/")
async def root():
    """Endpoint raíz para verificar el estado del servicio"""
    return {
        "status": "online", 
        "service": "DataInsight AI ML Service", 
        "version": "1.0.0"
    }

@app.get("/models")
async def get_available_models():
    """Retorna la lista de modelos disponibles"""
    return {
        "available_models": list(trained_models.keys()),
        "models_info": {
            name: {
                "feature_names": info.get("feature_names"),
                "target_names": info.get("target_names"),
                "accuracy": info.get("accuracy")
            } for name, info in trained_models.items()
        }
    }

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    """Realiza una predicción usando el modelo especificado"""
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")
    
    # Validar que el número de características coincida
    expected_features = len(trained_models[model_name].get("feature_names", []))
    if expected_features > 0 and len(request.features) != expected_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Número incorrecto de características. Se esperaban {expected_features}, pero se recibieron {len(request.features)}"
        )
    
    try:
        # Obtener el modelo
        model = trained_models[model_name]["model"]
        
        # Convertir características a numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        
        # Si el modelo soporta predict_proba, incluir las probabilidades
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features)[0].tolist()
        
        # Si hay nombres para las clases target, usar como predicción
        if "target_names" in trained_models[model_name] and trained_models[model_name]["target_names"] is not None:
            target_names = trained_models[model_name]["target_names"]
            prediction_label = target_names[prediction]
        else:
            prediction_label = prediction
        
        return {
            "prediction": prediction_label,
            "prediction_proba": prediction_proba,
            "model_name": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

@app.get("/datasets")
async def get_available_datasets():
    """Retorna la lista de datasets disponibles"""
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    available_datasets = []
    
    if os.path.exists(datasets_dir):
        available_datasets = [f for f in os.listdir(datasets_dir) if f.endswith((".csv", ".json"))]
    
    return {
        "available_datasets": available_datasets
    }

# Inicialización
@app.on_event("startup")
async def startup_event():
    """Inicializa los modelos cuando se inicia el servicio"""
    print("Inicializando servicio ML...")
    train_iris_model()
    print(f"Servicio ML inicializado con {len(trained_models)} modelos")

# Ejecutar el servidor si se ejecuta como script principal
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Iniciando servicio ML en http://{host}:{port}")
    uvicorn.run("ml_service:app", host=host, port=port, reload=True)