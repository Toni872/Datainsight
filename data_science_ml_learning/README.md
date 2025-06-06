# Proyecto de Aprendizaje de Ciencia de Datos y Machine Learning

Este directorio contiene material organizado para aprendizaje y desarrollo de proyectos en ciencia de datos y machine learning.

## Estructura del Proyecto

```
data_science_ml_learning/
├── 1_fundamentos/            # Conceptos básicos de Python para ciencia de datos
│   ├── introduccion.py       # Script introductorio a NumPy, Pandas y visualización
│   └── notebooks/            # Notebooks Jupyter con ejemplos
├── 2_estadistica_matematicas/# Fundamentos estadísticos y matemáticos
├── 3_machine_learning/       # Algoritmos y técnicas de machine learning
│   ├── no_supervisado/       # Clustering, PCA, etc.
│   └── supervisado/          # Clasificación, regresión, etc.
├── 4_deep_learning/          # Redes neuronales y deep learning
├── 5_especializacion/        # Áreas específicas de aplicación
│   ├── nlp/                  # Procesamiento del Lenguaje Natural
│   ├── series_temporales/    # Análisis de series temporales
│   ├── sistemas_recomendacion/# Sistemas de recomendación
│   └── vision_computador/    # Visión por computador
└── datasets/                # Conjuntos de datos para prácticas
```

## Requisitos

Los paquetes necesarios están listados en el archivo `requirements.txt`. Para instalar todas las dependencias, ejecuta:

```bash
pip install -r requirements.txt
```

## Primeros Pasos

1. Configura un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Comienza con los fundamentos:
   - Ejecuta `python 1_fundamentos/introduccion.py` para una introducción rápida
   - Explora los notebooks en la carpeta `1_fundamentos/notebooks/`

4. Avanza progresivamente por las carpetas según tu nivel de conocimiento

## Integraciones con la Aplicación Web

Este componente de ciencia de datos está diseñado para integrarse con la aplicación web TypeScript/Express del proyecto principal. Algunas posibles integraciones incluyen:

- API de predicciones basadas en modelos entrenados
- Visualizaciones de datos para el frontend
- Procesamiento de datos de usuarios para recomendaciones
- Análisis de sentimientos para comentarios o reseñas

## Contribuciones

Si deseas contribuir a este proyecto, por favor crea un fork y envía un pull request con tus cambios.

