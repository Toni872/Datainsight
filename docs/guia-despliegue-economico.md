# Guía de despliegue económico para DataInsight

Esta guía te ayudará a desplegar tu aplicación DataInsight con un presupuesto mínimo utilizando Vercel, MongoDB Atlas y Railway.

## Paso 1: Registrar un dominio económico

### Opción recomendada: Hostalia
1. Visita [Hostalia](https://www.hostalia.com/dominios/)
2. Busca disponibilidad para "datainsight.es" o "datainsight.com"
3. Registra el dominio (~10-15€/año, primer año puede ser gratuito con algunos planes)

## Paso 2: Configurar la base de datos MongoDB Atlas (gratuito)

1. Crea una cuenta en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Selecciona el plan "Free Tier" (M0)
3. Crea un nuevo cluster
4. Configura la conexión IP (Allow Access from Anywhere)
5. Crea un usuario de base de datos
6. Obtén la cadena de conexión

## Paso 3: Preparar el proyecto para Vercel

1. Crea un archivo `vercel.json` en la raíz del proyecto:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/app.ts",
      "use": "@vercel/node"
    },
    {
      "src": "public/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/js/(.*)",
      "dest": "/public/js/$1"
    },
    {
      "src": "/css/(.*)",
      "dest": "/public/css/$1"
    },
    {
      "src": "/img/(.*)",
      "dest": "/public/img/$1"
    },
    {
      "src": "/datasets/(.*)",
      "dest": "/public/datasets/$1"
    },
    {
      "src": "/(.*)\\.html",
      "dest": "/public/$1.html"
    },
    {
      "src": "/(favicon\\.ico|robots\\.txt|sitemap\\.xml)",
      "dest": "/public/$1"
    },
    {
      "src": "/api/(.*)",
      "dest": "/src/app.ts"
    },
    {
      "src": "/(.*)",
      "dest": "/src/app.ts"
    }
  ],
  "env": {
    "NODE_ENV": "production",
    "MONGODB_URI": "@mongodb_uri",
    "ML_SERVICE_URL": "@ml_service_url",
    "JWT_SECRET": "@jwt_secret"
  }
}
```

2. Modifica el archivo `package.json` para añadir configuración de Vercel:

```json
{
  "engines": {
    "node": "18.x"
  }
}
```

## Paso 4: Desplegar el servicio ML en Railway

1. Crea una cuenta en [Railway](https://railway.app/)
2. Conecta tu repositorio de GitHub
3. Selecciona el directorio `data_science_ml_learning`
4. Railway detectará automáticamente el Dockerfile
5. Configura las variables de entorno necesarias
6. Despliega el servicio y obtén la URL del API

## Paso 5: Desplegar la aplicación principal en Vercel

1. Crea una cuenta en [Vercel](https://vercel.com/signup)
2. Conecta tu repositorio de GitHub
3. Importa el proyecto
4. Configura las variables de entorno:
   - `MONGODB_URI`: URI de MongoDB Atlas
   - `ML_SERVICE_URL`: URL del servicio ML en Railway
   - `JWT_SECRET`: Genera una clave secreta para JWT
5. Despliega la aplicación

## Paso 6: Configurar el dominio personalizado

1. En Vercel, ve a "Settings" > "Domains"
2. Añade tu dominio (ej. datainsight.es)
3. Sigue las instrucciones para configurar los registros DNS en Hostalia
4. Vercel proporcionará automáticamente un certificado SSL

## Paso 7: Probar la aplicación

1. Accede a tu dominio personalizado
2. Verifica que todas las funcionalidades estén operativas
3. Prueba la conexión con el servicio ML

## Consideraciones

1. **Límites de uso gratuito**:
   - MongoDB Atlas: 512MB de almacenamiento
   - Vercel: Límites en funciones serverless
   - Railway: 5$ de crédito gratuito al mes

2. **Escalabilidad**:
   - Cuando necesites más recursos, puedes actualizar gradualmente cada servicio

3. **Respaldo de datos**:
   - Configura respaldos manuales periódicos para MongoDB Atlas

## Costos totales estimados

| Servicio | Plan | Costo mensual |
|----------|------|---------------|
| Dominio | Hostalia | ~1€/mes (amortizado) |
| MongoDB Atlas | M0 | 0€ |
| Vercel | Hobby | 0€ |
| Railway | Starter | ~5€/mes |
| **TOTAL** | | **~6€/mes** |

Esta configuración te permite tener una aplicación completa y funcional con un dominio personalizado por aproximadamente 6€ al mes, en lugar de los 130€ que costaría en Azure.
