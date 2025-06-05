/**
 * Versión simplificada para demostración
 */
const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Servir archivos estáticos
app.use(express.static(path.join(__dirname, '../public')));

// Middleware para JSON y form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Ruta API simple de planes para demostración
app.get('/api/subscription/plans', (req, res) => {
  const plans = [
    {
      id: 'free',
      name: 'Plan Gratuito',
      description: 'Perfecto para empezar y explorar la plataforma',
      price: {
        monthly: 0,
        yearly: 0
      },
      features: {
        apiCallsPerMonth: 100,
        modelTrainingPerMonth: 5,
        storageLimit: 50, // 50MB
        advancedModels: false,
        prioritySupport: false,
        customModels: false,
        dedicatedResources: false
      }
    },
    {
      id: 'basic',
      name: 'Plan Básico',
      description: 'Para pequeños equipos y proyectos iniciales',
      price: {
        monthly: 29.99,
        yearly: 299.99 // ~2 meses gratis
      },
      features: {
        apiCallsPerMonth: 1000,
        modelTrainingPerMonth: 20,
        storageLimit: 250, // 250MB
        advancedModels: true,
        prioritySupport: false,
        customModels: false,
        dedicatedResources: false
      },
      popular: true
    },
    {
      id: 'professional',
      name: 'Plan Profesional',
      description: 'Para equipos en crecimiento con necesidades avanzadas',
      price: {
        monthly: 79.99,
        yearly: 799.99 // ~2 meses gratis
      },
      features: {
        apiCallsPerMonth: 5000,
        modelTrainingPerMonth: 100,
        storageLimit: 1024, // 1GB
        advancedModels: true,
        prioritySupport: true,
        customModels: true,
        dedicatedResources: false
      }
    },
    {
      id: 'enterprise',
      name: 'Plan Empresarial',
      description: 'Recursos dedicados para grandes organizaciones',
      price: {
        monthly: 199.99,
        yearly: 1999.99 // ~2 meses gratis
      },
      features: {
        apiCallsPerMonth: 50000,
        modelTrainingPerMonth: 500,
        storageLimit: 10240, // 10GB
        advancedModels: true,
        prioritySupport: true,
        customModels: true,
        dedicatedResources: true
      }
    }
  ];

  res.json({
    success: true,
    plans
  });
});

// Ruta API simple para demo de usuario actual
app.get('/api/auth/me', (req, res) => {
  // Demo - simula usuario autenticado
  res.json({
    success: true,
    user: {
      _id: '60d21b4667d0d8992e610c85',
      name: 'Usuario Demo',
      email: 'demo@example.com',
      role: 'user',
      subscriptionPlan: 'basic',
      subscriptionStatus: 'active',
      subscriptionExpiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      apiCalls: { used: 450, lastResetDate: new Date() },
      modelTraining: { used: 12, lastResetDate: new Date() },
      storage: { used: 125 * 1024 * 1024, lastResetDate: new Date() }
    }
  });
});

// Ruta API simple para demo de suscripción actual
app.get('/api/subscription/current', (req, res) => {
  // Datos demo para mostrar
  res.json({
    success: true,
    subscription: {
      plan: 'basic',
      planName: 'Plan Básico',
      status: 'active',
      billingPeriod: 'monthly',
      expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      cancelAtPeriodEnd: false,
      quotaInfo: {
        apiCalls: {
          used: 450,
          limit: 1000,
          percentage: 45
        },
        modelTraining: {
          used: 12,
          limit: 20,
          percentage: 60
        },
        storage: {
          used: 125 * 1024 * 1024, // 125MB
          limit: 250 * 1024 * 1024, // 250MB
          percentage: 50,
          formattedUsed: '125 MB',
          formattedLimit: '250 MB'
        },
        supportedModelTiers: ['basic', 'advanced'],
        daysUntilReset: 12
      }
    }
  });
});

// Si ninguna ruta API coincide, devuelve la página principal
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Iniciar servidor
app.listen(PORT, () => {
  console.log(`Servidor simple ejecutándose en http://localhost:${PORT}`);
});
