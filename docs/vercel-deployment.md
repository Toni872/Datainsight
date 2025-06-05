# Implementación paso a paso de DataInsight en Vercel

Esta guía proporciona instrucciones específicas para implementar la aplicación DataInsight utilizando Vercel, MongoDB Atlas y Railway con un presupuesto mínimo.

## Paso 1: Crear archivo vercel.json

Crea este archivo en la raíz del proyecto:

```bash
# Ejecuta desde la raíz del proyecto
cd "c:\Users\Toninopc\Desktop\Programacion\mi-proyecto"
```

## Paso 2: Adaptar app.ts para entorno Vercel

Vercel funciona mejor con aplicaciones que exportan una instancia de Express. Modifica ligeramente tu app.ts:

```typescript
// Al final del archivo app.ts, añade:
export default app;
```

## Paso 3: Crear base de datos en MongoDB Atlas

1. Regístrate en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Crea un cluster gratuito (M0)
3. Configura acceso de red (permitir desde cualquier lugar para desarrollo)
4. Crea un usuario de base de datos
5. Obtén la cadena de conexión (URI)

## Paso 4: Desplegar servicio ML en Railway

1. Regístrate en [Railway](https://railway.app/)
2. Crea un nuevo proyecto
3. Selecciona "Deploy from GitHub repo"
4. Configura el directorio: `data_science_ml_learning`
5. Railway detectará automáticamente el Dockerfile
6. Añade variables de entorno según sea necesario
7. Despliega y copia la URL resultante

## Paso 5: Desplegar en Vercel

1. Instala Vercel CLI:
```bash
npm i -g vercel
```

2. Inicia sesión en Vercel:
```bash
vercel login
```

3. Configura las variables de entorno:
```bash
vercel env add MONGODB_URI
# Pega la URI de MongoDB Atlas cuando se solicite

vercel env add ML_SERVICE_URL
# Pega la URL del servicio ML de Railway

vercel env add JWT_SECRET
# Genera un valor aleatorio seguro, por ejemplo usando: openssl rand -base64 32
```

4. Despliega el proyecto:
```bash
vercel --prod
```

5. Configura tu dominio personalizado:
```bash
vercel domains add datainsight.es
```

## Paso 6: Migrar datos (si es necesario)

Si tienes datos existentes:

1. Exporta datos desde tu entorno local:
```bash
mongodump --uri="mongodb://localhost:27017/datainsight"
```

2. Importa datos a MongoDB Atlas:
```bash
mongorestore --uri="<tu-uri-de-mongodb-atlas>" dump/
```

## Paso 7: Configurar Github Actions para CI/CD (opcional)

Crea un archivo `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Vercel

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
```

## Costos totales

| Servicio | Plan | Costo |
|----------|------|-------|
| Vercel | Hobby | 0€/mes |
| MongoDB Atlas | M0 | 0€/mes |
| Railway | Starter | ~5€/mes |
| Dominio | Hostalia | ~10-15€/año |
| **TOTAL** | | **~6€/mes** |

## Ventajas de esta configuración

1. **Despliegue simplificado**: Implementación basada en Git
2. **Escalabilidad**: Fácil actualización a planes superiores si crece el tráfico
3. **SSL gratuito**: Certificados automáticos con Vercel
4. **CI/CD incorporado**: Despliegue automático al hacer push a GitHub
5. **Ahorro significativo**: ~124€/mes menos que la opción de Azure

## Limitaciones

1. **Recursos limitados**: Adecuado para desarrollo y tráfico bajo-medio
2. **Tiempo de actividad**: No hay SLA en los planes gratuitos
3. **Soporte**: Limitado o por comunidad

Esta configuración es ideal para validar tu proyecto y comenzar a adquirir usuarios con un costo mínimo. Cuando el proyecto genere ingresos o necesite más recursos, puedes actualizar gradualmente a planes superiores.
