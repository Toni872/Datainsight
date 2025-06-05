# Presupuesto estimado para DataInsight AI

## Costos iniciales (primer año)

| Concepto | Proveedor | Costo estimado | Notas |
|----------|-----------|----------------|-------|
| Dominio datainsight.ai | Namecheap | 35-40€/año | Los dominios .ai suelen ser más caros que los comunes |
| Certificado SSL | Let's Encrypt / Azure | 0€ | Gratuito con Azure App Service |
| **Total costos iniciales** | | **35-40€** | |

## Costos mensuales (infraestructura Azure)

| Servicio | Nivel | Costo mensual | Notas |
|----------|-------|---------------|-------|
| App Service | Plan B1 (1 núcleo, 1.75GB RAM) | 45€ | Para la aplicación web Node.js |
| Container App | 1 CPU, 2GB RAM | 40€ | Para el servicio ML Python |
| Cosmos DB | API MongoDB, 10GB | 25€ | Base de datos principal |
| Storage Account | Standard LRS, 50GB | 5€ | Almacenamiento de archivos |
| CDN | Standard | 15€ | Distribución de contenido estático |
| Key Vault | Estándar | 2€ | Gestión de secretos |
| **Total costos mensuales** | | **132€** | |

## Costos anuales

| Concepto | Costo anual |
|----------|-------------|
| Infraestructura Azure | 1.584€ (132€ x 12 meses) |
| Dominio | 35-40€ |
| **Total anual** | **1.619-1.624€** |

## Opciones para reducir costos

### Opción 1: Plan de desarrollo/pruebas
- App Service: Plan B1 compartido (25€/mes)
- Container App: Mínimo (20€/mes)
- Cosmos DB: Nivel compartido (10€/mes)
- **Total reducido**: ~70€/mes (840€/año)

### Opción 2: Proveedor alternativo (DigitalOcean)
- Droplet (4GB): 24€/mes - Para toda la aplicación
- MongoDB Atlas (M10): 45€/mes - Base de datos gestionada
- Spaces (S3): 5€/mes - Almacenamiento
- **Total**: ~74€/mes (888€/año)

### Opción 3: Azure con reservas a un año
Si pagas por adelantado con un compromiso anual, puedes obtener descuentos de hasta un 20-30% en varios servicios de Azure.

## Consideraciones adicionales

- **Escalado automático**: Los servicios se pueden configurar para escalar automáticamente según la carga, lo que puede aumentar los costos en periodos de alta demanda.
- **Ancho de banda**: Los costos estimados incluyen un uso moderado de ancho de banda. Si tu aplicación genera mucho tráfico, estos costos podrían aumentar.
- **Almacenamiento**: A medida que acumules más datos, los costos de almacenamiento aumentarán.
- **Backup y DR**: La configuración de copias de seguridad y recuperación ante desastres puede añadir aproximadamente un 20% adicional al costo.

## Recomendación

Para una empresa en fase inicial, recomendamos comenzar con la configuración estándar y monitorear el uso durante los primeros 3 meses. Después, podemos optimizar los recursos según los patrones de uso reales y potencialmente reducir costos.
