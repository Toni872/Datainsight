# Plataforma Avanzada de Machine Learning

Un proyecto combinado de aplicación web TypeScript/Express.js y ciencia de datos con Python que implementa algoritmos avanzados de machine learning, optimizaciones de rendimiento y visualizaciones interactivas.

## Aviso Legal y Licencia

**IMPORTANTE**: Este proyecto está protegido bajo la licencia GNU Affero General Public License v3.0 (AGPL-3.0). Cualquier uso, modificación o distribución debe cumplir con los términos de esta licencia, que incluye:

- El código fuente completo de cualquier trabajo derivado debe estar disponible bajo la misma licencia
- Se debe mantener el aviso de copyright y atribución al autor original
- Cualquier versión modificada debe indicar claramente que se han realizado cambios

Copyright (c) 2025 [Tu Nombre]. Todos los derechos reservados.

## Descripción General

Este proyecto implementa una plataforma completa para análisis de datos y machine learning que incluye:

- Algoritmos avanzados de machine learning supervisado y no supervisado
- Sistema de optimización y evaluación de modelos
- Análisis de series temporales
- Visualizaciones interactivas con sistema de caché
- API REST para integración con aplicaciones web
- Interfaz web para gestión y visualización de modelos

## Estructura del Proyecto

El proyecto está dividido en dos componentes principales:

### Componente Web (TypeScript/Express.js)

```
├── src/                     # Código fuente de la aplicación web
│   ├── app.ts               # Punto de entrada de la aplicación
│   ├── controllers/         # Controladores
│   ├── middlewares/         # Middlewares y funciones de utilidad
│   ├── models/              # Modelos de datos
│   ├── routes/              # Definiciones de rutas
│   └── utils/               # Utilidades y funciones auxiliares
├── public/                  # Archivos estáticos (CSS, JS, imágenes)
│   ├── css/
│   └── js/
└── tests/                   # Pruebas unitarias y de integración
```

### Componente de Ciencia de Datos (Python)

```
├── data_science_ml_learning/
    ├── 1_fundamentos/       # Conceptos básicos para ciencia de datos
    ├── 2_estadistica_matematicas/
    ├── 3_machine_learning/
    ├── 4_deep_learning/
    ├── 5_especializacion/
    └── datasets/            # Conjuntos de datos para prácticas
```

## Requisitos

### Para el componente web:
```bash
npm install
```

### Para el componente de ciencia de datos:
```bash
pip install -r data_science_ml_learning/requirements.txt
```

## Ejecución

### Componente web:
```bash
npm start
```

### Componente de ciencia de datos:
Consulta el archivo `data_science_ml_learning/README.md` para obtener más detalles.

## Desarrollo

Este proyecto está diseñado para demostrar la integración de una aplicación web moderna con capacidades de ciencia de datos y machine learning.

### Características planificadas:

1. **Aplicación web**:
   - API RESTful
   - Sistema de autenticación
   - Panel de administración
   - Visualizaciones interactivas

2. **Ciencia de datos**:
   - Modelos de machine learning
   - Análisis predictivo
   - Procesamiento de lenguaje natural
   - Visualización avanzada de datos

## Contribuciones

Las contribuciones son bienvenidas. Por favor, crea un fork del proyecto y envía un pull request con tus cambios.

## Configuración SSL (HTTPS)

### Desarrollo local con certificados autofirmados

Para ejecutar la aplicación con HTTPS durante el desarrollo local:

1. Generar certificados SSL autofirmados:

   ```bash
   npm run generate-cert
   ```
   Esto creará los archivos `privkey.pem` y `fullchain.pem` en la carpeta `ssl/`.

2. Iniciar el servidor con SSL habilitado:

   ```bash
   npm run dev:ssl
   ```

3. Aceptar el certificado autofirmado en el navegador (aparecerá una advertencia de seguridad).

### Producción con Let's Encrypt

Para configurar HTTPS en producción con certificados válidos:

1. Instala Win-ACME (para Windows/IIS) desde: 
   [https://github.com/win-acme/win-acme/releases](https://github.com/win-acme/win-acme/releases)

2. Ejecuta Win-ACME y sigue el asistente para obtener un certificado para tu dominio.

3. Actualiza las rutas de los certificados en tu archivo `.env`:

   ```env
   SSL_ENABLED=true
   SSL_KEY_PATH=/ruta/a/privkey.pem
   SSL_CERT_PATH=/ruta/a/fullchain.pem
   ```

4. Inicia la aplicación en modo producción:

   ```bash
   npm run production
   ```

5. Configuración adicional IIS: Asegúrate de que el binding HTTPS esté configurado en IIS con el puerto 443.

## Licencia

ISC