FROM python:3.10-slim-bullseye

WORKDIR /app

# Copiar los archivos de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Instalar FastAPI para la API REST
RUN pip install fastapi uvicorn

# Crear una aplicación FastAPI básica para servir modelos ML
RUN echo 'from fastapi import FastAPI, HTTPException\n\
import numpy as np\n\
import pandas as pd\n\
import pickle\n\
import os\n\
from pydantic import BaseModel\n\
\n\
app = FastAPI(title="DataInsight AI ML Service")\n\
\n\
class PredictionRequest(BaseModel):\n\
    features: list\n\
\n\
@app.get("/")\n\
def root():\n\
    return {"message": "DataInsight AI ML Service API"}\n\
\n\
@app.get("/models")\n\
def list_models():\n\
    # Enumerar los modelos disponibles\n\
    models_dir = os.path.join("3_machine_learning", "supervisado", "models")\n\
    os.makedirs(models_dir, exist_ok=True)\n\
    models = [f.replace(".pkl", "") for f in os.listdir(models_dir) if f.endswith(".pkl")]\n\
    return {"available_models": models}\n\
\n\
@app.post("/predict/{model_name}")\n\
def predict(model_name: str, request: PredictionRequest):\n\
    model_path = os.path.join("3_machine_learning", "supervisado", "models", f"{model_name}.pkl")\n\
    \n\
    # Verificar si el modelo existe\n\
    if not os.path.exists(model_path):\n\
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")\n\
    \n\
    # Cargar el modelo\n\
    try:\n\
        with open(model_path, "rb") as f:\n\
            model = pickle.load(f)\n\
    except Exception as e:\n\
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")\n\
    \n\
    # Convertir la entrada a numpy array\n\
    try:\n\
        features = np.array(request.features).reshape(1, -1)\n\
        prediction = model.predict(features).tolist()\n\
        probability = None\n\
        \n\
        # Intentar obtener probabilidades si el modelo lo soporta\n\
        try:\n\
            if hasattr(model, "predict_proba"):\n\
                probability = model.predict_proba(features).tolist()\n\
        except:\n\
            pass\n\
            \n\
        return {\n\
            "model": model_name,\n\
            "prediction": prediction,\n\
            "probability": probability\n\
        }\n\
    except Exception as e:\n\
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")\n\
\n\
@app.get("/datasets")\n\
def list_datasets():\n\
    # Enumerar los datasets disponibles\n\
    datasets_dir = "datasets"\n\
    if not os.path.exists(datasets_dir):\n\
        return {"available_datasets": []}\n\
    \n\
    datasets = [f for f in os.listdir(datasets_dir) if f.endswith((".csv", ".json"))]\n\
    return {"available_datasets": datasets}\n' > main.py

# Exponer el puerto para la API
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]