# DataInsight - Despliegue en Vercel

Este documento contiene instrucciones detalladas para desplegar DataInsight en Vercel como una alternativa económica a Azure.

## Requisitos previos

1. Cuenta en [Vercel](https://vercel.com)
2. Cuenta en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
3. Cuenta en [Railway](https://railway.app) (para el servicio ML)
4. Node.js 18.x instalado
5. npm instalado

## Pasos para el despliegue

### 1. Preparar MongoDB Atlas

1. Crea un cluster gratuito (M0)
2. Configura un usuario de base de datos
3. Permite acceso desde cualquier IP (0.0.0.0/0)
4. Obtén la cadena de conexión

### 2. Preparar el servicio ML en Railway

1. Conecta tu repositorio de GitHub
2. Selecciona el directorio `data_science_ml_learning`
3. Railway detectará automáticamente el Dockerfile
4. Despliega y obtén la URL del API

### 3. Configurar variables de entorno

Crea un archivo `.env` basado en `.env.example` con:

```
MONGODB_URI=tu-uri-de-mongodb
ML_SERVICE_URL=tu-url-de-railway
JWT_SECRET=genera-una-clave-segura
```

### 4. Desplegar a Vercel

Puedes usar el script automatizado:

```powershell
./deploy-to-vercel.ps1
```

O manualmente:

1. Instala Vercel CLI: `npm i -g vercel`
2. Inicia sesión: `vercel login`
3. Configura variables: `vercel env add NOMBRE_VARIABLE`
4. Despliega: `vercel --prod`
5. Configura dominio: `vercel domains add tu-dominio.es`

### 5. Verificar el despliegue

1. Accede a la URL proporcionada por Vercel
2. Verifica la conexión con MongoDB y el servicio ML
3. Prueba todas las funcionalidades, especialmente la autenticación

## Costos estimados

| Servicio | Plan | Costo |
|----------|------|-------|
| Vercel | Hobby | 0€/mes |
| MongoDB Atlas | M0 | 0€/mes |
| Railway | Starter | ~5€/mes |
| Dominio | - | ~10-15€/año |
| **TOTAL** | | **~6€/mes** |

## Limitaciones

- **MongoDB Atlas M0**: Limitado a 512MB de almacenamiento
- **Vercel Hobby**: Funciones serverless limitadas en ejecución
- **Railway Starter**: Crédito limitado mensual

## Solución de problemas

- **Error en conexión a MongoDB**: Verifica la cadena de conexión y los permisos de IP
- **Error en servicio ML**: Asegúrate de que el contenedor en Railway está activo
- **CORS no funciona**: Verifica la configuración en app.ts

## Documentación adicional

Para información más detallada, consulta:
- [docs/vercel-deployment.md](docs/vercel-deployment.md)
- [docs/opciones-economicas.md](docs/opciones-economicas.md)
- [docs/guia-despliegue-economico.md](docs/guia-despliegue-economico.md)
