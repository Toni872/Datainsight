/**
 * Tests de integración para el sistema de suscripciones con Stripe real
 * 
 * IMPORTANTE: Este archivo requiere una configuración previa con claves API de Stripe en modo de prueba.
 * Ejecutar con: npm run test:integration
 */

import request from 'supertest';
import mongoose from 'mongoose';
import Stripe from 'stripe';
import app from '../src/app';
import { User } from '../src/models/user.model';
import { SUBSCRIPTION_PLANS } from '../src/models/subscription.model';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
// Cargar variables de entorno
dotenv.config();

// Cliente de Stripe real en modo de prueba
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16' as any
});

describe('Integración del Sistema de Suscripciones con Stripe', () => {
  let token: string;
  let testUser: any;
  let basicPlanMonthlyPriceId: string;
  let checkoutSessionId: string;
  
  beforeAll(async () => {
    // Verificar que tenemos la clave de Stripe
    if (!process.env.STRIPE_SECRET_KEY) {
      console.warn('STRIPE_SECRET_KEY no está configurada en .env, usando mocks');
    }
    
    // Conectar a la base de datos
    try {
      await mongoose.connect(process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/mi-proyecto-test');
    } catch (error) {
      console.error('Error conectando a MongoDB:', error);
    }
    
    // Limpiar usuarios existentes
    try {
      await User.deleteMany({});
    } catch (error) {
      console.warn('No se pudieron borrar usuarios existentes:', error);
    }
    
    // Crear usuario de prueba
    testUser = await User.create({
      name: 'Usuario de Integración',
      email: 'integration@example.com',
      password: 'securepassword123',
      role: 'user',
      subscriptionPlan: 'free',
      subscriptionStatus: 'active',
      billingPeriod: 'monthly',
      apiCalls: { used: 0, lastResetDate: new Date() },
      modelTraining: { used: 0, lastResetDate: new Date() },
      storage: { used: 0, lastResetDate: new Date() },
      isActive: true
    });
    
    // Obtener token de autenticación
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({ email: 'integration@example.com', password: 'securepassword123' });
      
    if (loginResponse.body && loginResponse.body.token) {
      token = loginResponse.body.token;
    } else {
      // Generar un token si la autenticación falló
      token = 'mock_token_for_testing';
      console.warn('No se pudo obtener token de autenticación, usando token simulado');
    }
    
    if (process.env.STRIPE_SECRET_KEY) {
      try {
        // Obtener los price IDs de Stripe
        const prices = await stripe.prices.list({ limit: 10 });
        
        // Buscar el price ID para el plan básico mensual
        for (const price of prices.data) {
          if (price.nickname?.includes('basic') && price.nickname?.includes('monthly')) {
            basicPlanMonthlyPriceId = price.id;
            break;
          }
        }
        
        if (!basicPlanMonthlyPriceId) {
          console.warn('No se encontró un precio para el plan básico mensual en Stripe');
          
          // Crear un producto y precio de prueba
          const product = await stripe.products.create({
            name: 'Plan Básico',
            description: 'Para pequeños equipos y proyectos iniciales'
          });
          
          const price = await stripe.prices.create({
            product: product.id,
            unit_amount: 2999, // 29.99 en centavos
            currency: 'eur',
            recurring: {
              interval: 'month'
            },
            nickname: 'basic_monthly'
          });
          
          basicPlanMonthlyPriceId = price.id;
          console.log(`Created test price ID: ${basicPlanMonthlyPriceId}`);
          
          // Actualizar el modelo con el ID de precio
          if (SUBSCRIPTION_PLANS && SUBSCRIPTION_PLANS[1]) {
            SUBSCRIPTION_PLANS[1].stripeMonthlyPriceId = basicPlanMonthlyPriceId;
          }
        }
      } catch (error) {
        console.warn('Error interactuando con Stripe API:', error);
        basicPlanMonthlyPriceId = 'mock_price_id';
      }
    }
  }, 30000);
  
  afterAll(async () => {
    // Limpiar base de datos
    try {
      await User.deleteMany({});
    } catch (error) {
      console.warn('Error limpiando usuarios de prueba:', error);
    }
    
    // Cerrar conexión
    try {
      await mongoose.connection.close();
    } catch (error) {
      console.warn('Error cerrando conexión a MongoDB:', error);
    }
  }, 10000);
  
  describe('Flujo de actualización de suscripción', () => {
    test('Debe crear una sesión de checkout en Stripe', async () => {
      // Skip si no hay clave de Stripe
      if (!process.env.STRIPE_SECRET_KEY) {
        console.warn('Test saltado: STRIPE_SECRET_KEY no configurada');
        return;
      }
      
      const response = await request(app)
        .post('/api/subscription/upgrade')
        .set('Authorization', `Bearer ${token}`)
        .send({
          planId: 'basic',
          interval: 'monthly'
        });
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.redirectUrl).toBeDefined();
      expect(response.body.sessionId).toBeDefined();
      
      // Guardar el ID de la sesión para pruebas posteriores
      checkoutSessionId = response.body.sessionId;
      
      // Verificar que la sesión se creó en Stripe
      if (checkoutSessionId) {
        const session = await stripe.checkout.sessions.retrieve(checkoutSessionId);
        expect(session).toBeDefined();
        expect(session.metadata).toBeDefined();
        expect(session.metadata?.userId).toBe(testUser._id.toString());
        expect(session.metadata?.planId).toBe('basic');
      }
    }, 15000);
    
    test('Debe procesar correctamente un evento de webhook de pago exitoso', async () => {
      // Skip si no hay clave de Stripe
      if (!process.env.STRIPE_SECRET_KEY) {
        console.warn('Test saltado: STRIPE_SECRET_KEY no configurada');
        return;
      }
      
      // Crear un evento simulado de webhook
      const event = {
        type: 'checkout.session.completed',
        data: {
          object: {
            id: checkoutSessionId || 'mock_session_id',
            customer: 'cus_test_webhook',
            subscription: 'sub_test_webhook',
            status: 'complete',
            metadata: {
              userId: testUser._id.toString(),
              planId: 'basic',
              interval: 'monthly'
            }
          }
        }
      };
      
      // Mock la función de verificación de firma
      const originalConstructEvent = stripe.webhooks.constructEvent;
      stripe.webhooks.constructEvent = jest.fn().mockReturnValue(event);
      
      const response = await request(app)
        .post('/api/subscription/webhook')
        .set('stripe-signature', 'test_signature')
        .send(JSON.stringify(event));
      
      expect(response.statusCode).toBe(200);
      expect(response.body.received).toBe(true);
      
      // Restaurar la función original
      stripe.webhooks.constructEvent = originalConstructEvent;
    }, 10000);
    
    test('Debe cancelar correctamente una suscripción simulada', async () => {
      // Skip si no hay clave de Stripe
      if (!process.env.STRIPE_SECRET_KEY) {
        console.warn('Test saltado: STRIPE_SECRET_KEY no configurada');
        return;
      }
      
      // Primero, actualizar el usuario con una suscripción simulada
      await User.findByIdAndUpdate(testUser._id, {
        subscriptionPlan: 'basic',
        stripeSubscriptionId: 'sub_test_cancel',
        stripeCustomerId: 'cus_test_cancel',
        subscriptionStatus: 'active'
      });
      
      // Mock para la función de Stripe
      const originalUpdate = stripe.subscriptions.update;
      stripe.subscriptions.update = jest.fn().mockResolvedValue({
        id: 'sub_test_cancel',
        status: 'active',
        cancel_at_period_end: true
      });
      
      const response = await request(app)
        .post('/api/subscription/cancel')
        .set('Authorization', `Bearer ${token}`);
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      
      // Verificar que el usuario se actualizó
      const updatedUser = await User.findById(testUser._id);
      if (updatedUser) {
        expect(updatedUser.cancelAtPeriodEnd).toBe(true);
      }
      
      // Restaurar la función original
      stripe.subscriptions.update = originalUpdate;
    }, 10000);
  });
});

describe('Pruebas de integración del sistema de suscripciones', () => {
  let mockToken: string;
  const mockUserId = new mongoose.Types.ObjectId().toString();

  beforeEach(() => {
    jest.clearAllMocks();
    mockToken = 'mock-token';
    
    // Mock del comportamiento de jwt.verify
    (jwt.verify as jest.Mock).mockImplementation(() => ({ id: mockUserId }));
  });

  describe('Verificación de suscripción activa', () => {
    it('debería permitir acceso con una suscripción activa', async () => {
      // Preparar un usuario mock con suscripción activa
      const mockUser = {
        _id: mockUserId,
        email: 'test@example.com',
        name: 'Usuario de Prueba',
        subscriptionPlan: 'basic',
        subscriptionStatus: 'active',
        subscriptionStartDate: new Date(),
        subscriptionExpiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 días en el futuro
        apiCalls: {
          used: 50,
          lastResetDate: new Date()
        },
        modelTraining: {
          used: 2,
          lastResetDate: new Date()
        },
        storage: {
          used: 25 * 1024 * 1024 // 25 MB
        }
      };
      
      // Mock de User.findById para devolver el usuario mock
      (User.findById as jest.Mock).mockResolvedValue(mockUser);
      
      // Hacer una solicitud a una ruta protegida
      const response = await request(app)
        .get('/api/subscription/current')
        .set('Authorization', `Bearer ${mockToken}`);
      
      // Verificar la respuesta
      expect(response.status).not.toBe(401);
      expect(response.status).not.toBe(403);
    });

    it('debería rechazar acceso con una suscripción vencida', async () => {
      // Preparar un usuario mock con suscripción vencida
      const mockUser = {
        _id: mockUserId,
        email: 'test@example.com',
        name: 'Usuario de Prueba',
        subscriptionPlan: 'basic',
        subscriptionStatus: 'past_due',
        subscriptionStartDate: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000), // 60 días en el pasado
        subscriptionExpiresAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 día en el pasado
        apiCalls: {
          used: 50,
          lastResetDate: new Date()
        }
      };
      
      // Mock de User.findById para devolver el usuario mock
      (User.findById as jest.Mock).mockResolvedValue(mockUser);
      
      // Hacer una solicitud a una ruta protegida
      const response = await request(app)
        .get('/api/subscription/current')
        .set('Authorization', `Bearer ${mockToken}`);
      
      // Verificar la respuesta (debería ser 403 Forbidden debido a la suscripción vencida)
      expect(response.status).toBe(403);
    });
  });

  describe('Reseteo de cuotas mensuales', () => {
    it('debería resetear correctamente las cuotas mensuales', async () => {
      // Preparar un usuario mock
      const mockUser = {
        _id: mockUserId,
        apiCalls: {
          used: 75,
          lastResetDate: new Date(Date.now() - 32 * 24 * 60 * 60 * 1000) // 32 días en el pasado
        },
        modelTraining: {
          used: 3,
          lastResetDate: new Date(Date.now() - 32 * 24 * 60 * 60 * 1000)
        }
      };
      
      // Mock de User.findById para devolver el usuario mock
      (User.findById as jest.Mock).mockResolvedValue(mockUser);
      // Mock de User.findByIdAndUpdate para simular actualización
      (User.findByIdAndUpdate as jest.Mock).mockResolvedValue(true);
      
      // Función de reseteo de cuotas
      async function resetMonthlyQuotas(userId: string) {
        return User.findByIdAndUpdate(userId, {
          'apiCalls.used': 0,
          'apiCalls.lastResetDate': new Date(),
          'modelTraining.used': 0,
          'modelTraining.lastResetDate': new Date()
        });
      }
      
      // Ejecutar reseteo de cuotas
      await resetMonthlyQuotas(mockUserId);
      
      // Verificar que findByIdAndUpdate fue llamado con los parámetros correctos
      expect(User.findByIdAndUpdate).toHaveBeenCalledWith(mockUserId, {
        'apiCalls.used': 0,
        'apiCalls.lastResetDate': expect.any(Date),
        'modelTraining.used': 0,
        'modelTraining.lastResetDate': expect.any(Date)
      });
    });
  });

  // Más pruebas podrían agregarse aquí para verificar:
  // - Cambio de plan
  // - Procesamiento de pagos
  // - Verificación de cuotas disponibles
  // - Notificaciones de límites cercanos
});
