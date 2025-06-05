# Comandos para despliegue manual en Vercel

# 1. Iniciar sesión en Vercel
vercel login

# 2. Configurar variables de entorno esenciales
vercel env add MONGODB_URI
vercel env add JWT_SECRET
vercel env add ML_SERVICE_URL

# 3. Desplegar a producción
vercel --prod

# 4. Configurar dominio personalizado (opcional)
vercel domains add tu-dominio.es
