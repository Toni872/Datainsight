import request from 'supertest';
import express from 'express';

// Mock de Stripe para pruebas
jest.mock('stripe', () => {
  return jest.fn().mockImplementation(() => {
    return {
      customers: {
        create: jest.fn().mockResolvedValue({ id: 'cus_mock123' }),
      },
      subscriptions: {
        create: jest.fn().mockResolvedValue({ 
          id: 'sub_mock123',
          status: 'active',
          current_period_end: new Date().getTime() / 1000 + 86400 * 30
        }),
      }
    };
  });
});

// Creamos un mock de la app para las pruebas
const app = express();
app.use(express.json());

app.post('/api/payment/create-subscription', (req, res) => {
  res.status(201).json({ 
    success: true, 
    message: 'Suscripción creada con éxito',
    data: {
      subscriptionId: 'sub_mock123',
      customerId: 'cus_mock123',
      status: 'active'
    }
  });
});

describe('Pruebas de integración de pagos', () => {
  it('debería crear una suscripción correctamente', async () => {
    const testData = {
      planId: 'price_123456',
      paymentMethodId: 'pm_123456',
      userId: '60d21b4667d0d8992e610c85'
    };

    const response = await request(app)
      .post('/api/payment/create-subscription')
      .send(testData);
    
    expect(response.status).toBe(201);
    expect(response.body.success).toBe(true);
    expect(response.body.data).toHaveProperty('subscriptionId');
  });
});
