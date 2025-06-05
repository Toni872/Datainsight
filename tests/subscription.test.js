/**
 * Tests para el sistema de suscripciones
 * Ejecutar con: npm test -- --testPathPattern=subscription.test.js
 */

const request = require('supertest');
const mongoose = require('mongoose');
const Stripe = require('stripe');
const app = require('../src/app').default;
const { User } = require('../src/models/user.model');

// Mock de Stripe
jest.mock('stripe');

describe('Sistema de Suscripciones', () => {
  let token;
  let userId;
  
  beforeAll(async () => {
    // Conectar a la base de datos de prueba
    await mongoose.connect(process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/mi-proyecto-test');
    
    // Crear un usuario de prueba
    const testUser = await User.create({
      name: 'Usuario de Prueba',
      email: 'test@example.com',
      password: 'password123',
      role: 'user',
      subscriptionPlan: 'free',
      subscriptionStatus: 'active',
      cancelAtPeriodEnd: false,
      billingPeriod: 'monthly',
      apiCalls: {
        used: 0,
        lastResetDate: new Date()
      },
      modelTraining: {
        used: 0,
        lastResetDate: new Date()
      },
      storage: {
        used: 0,
        lastResetDate: new Date()
      },
      isActive: true
    });
    
    userId = testUser._id.toString();
    
    // Login para obtener token
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'password123' });
      
    token = loginResponse.body.token;
    
    // Configurar Mock de Stripe
    Stripe.mockImplementation(() => ({
      checkout: {
        sessions: {
          create: jest.fn().mockResolvedValue({
            id: 'cs_test_123',
            url: 'https://stripe.com/checkout/cs_test_123',
            status: 'open'
          }),
          retrieve: jest.fn().mockResolvedValue({
            id: 'cs_test_123',
            status: 'complete',
            customer: 'cus_test123',
            subscription: 'sub_test123',
            metadata: {
              userId,
              planId: 'basic',
              interval: 'monthly'
            }
          })
        }
      },
      subscriptions: {
        retrieve: jest.fn().mockResolvedValue({
          id: 'sub_test123',
          status: 'active',
          current_period_start: Math.floor(Date.now() / 1000),
          current_period_end: Math.floor(Date.now() / 1000) + 30 * 24 * 60 * 60
        }),
        update: jest.fn().mockResolvedValue({
          id: 'sub_test123',
          status: 'active',
          cancel_at_period_end: true
        }),
        cancel: jest.fn().mockResolvedValue({
          id: 'sub_test123',
          status: 'canceled'
        })
      },
      invoices: {
        list: jest.fn().mockResolvedValue({
          data: [{
            id: 'in_test123',
            amount_paid: 2999,
            currency: 'eur',
            status: 'paid',
            created: Math.floor(Date.now() / 1000),
            period_start: Math.floor(Date.now() / 1000) - 30 * 24 * 60 * 60,
            period_end: Math.floor(Date.now() / 1000),
            invoice_pdf: 'https://stripe.com/invoice.pdf'
          }]
        })
      },
      webhooks: {
        constructEvent: jest.fn().mockReturnValue({
          type: 'invoice.paid',
          data: {
            object: {
              id: 'in_test123',
              customer: 'cus_test123',
              subscription: 'sub_test123',
              amount_paid: 2999,
              currency: 'eur',
              period_start: Math.floor(Date.now() / 1000) - 30 * 24 * 60 * 60,
              period_end: Math.floor(Date.now() / 1000)
            }
          }
        })
      }
    }));
  });
  
  afterAll(async () => {
    // Limpiar la base de datos
    await User.deleteMany({});
    await mongoose.connection.close();
  });
  
  describe('API de Suscripciones', () => {
    test('Debe obtener la suscripción actual del usuario', async () => {
      const response = await request(app)
        .get('/api/subscription/current')
        .set('Authorization', `Bearer ${token}`);
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.subscription).toBeDefined();
      expect(response.body.subscription.plan).toBe('free');
    });
    
    test('Debe listar todos los planes disponibles', async () => {
      const response = await request(app)
        .get('/api/subscription/plans');
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.plans).toBeDefined();
      expect(Array.isArray(response.body.plans)).toBe(true);
      expect(response.body.plans.length).toBeGreaterThan(0);
    });
    
    test('Debe iniciar un proceso de actualización de suscripción', async () => {
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
    });
    
    test('Debe procesar correctamente una nueva suscripción', async () => {
      const response = await request(app)
        .post('/api/subscription/process')
        .set('Authorization', `Bearer ${token}`)
        .send({
          sessionId: 'cs_test_123'
        });
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.message).toBeDefined();
      
      // Verificar que el usuario se ha actualizado
      const updatedUser = await User.findById(userId);
      expect(updatedUser.subscriptionPlan).toBe('basic');
      expect(updatedUser.subscriptionStatus).toBe('active');
    });
    
    test('Debe cancelar correctamente una suscripción', async () => {
      // Primero asegurarse de que el usuario tiene una suscripción de pago
      await User.findByIdAndUpdate(userId, {
        subscriptionPlan: 'basic',
        stripeSubscriptionId: 'sub_test123'
      });
      
      const response = await request(app)
        .post('/api/subscription/cancel')
        .set('Authorization', `Bearer ${token}`);
      
      expect(response.statusCode).toBe(200);
      expect(response.body.success).toBe(true);
      
      // Verificar que el usuario se ha actualizado
      const updatedUser = await User.findById(userId);
      expect(updatedUser.cancelAtPeriodEnd).toBe(true);
    });
    
    test('Debe procesar correctamente un webhook de Stripe', async () => {
      const response = await request(app)
        .post('/api/subscription/webhook')
        .set('stripe-signature', 'test_signature')
        .send(JSON.stringify({
          type: 'invoice.paid',
          data: {
            object: {
              id: 'in_test123',
              customer: 'cus_test123',
              subscription: 'sub_test123',
              amount_paid: 2999,
              currency: 'eur'
            }
          }
        }));
      
      expect(response.statusCode).toBe(200);
      expect(response.body.received).toBe(true);
    });
  });
});
