import request from 'supertest';
import express from 'express';

// Creamos un mock de la app para las pruebas
const app = express();
app.get('/', (req, res) => {
  res.status(200).send('OK');
});

app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

describe('Pruebas de la aplicación', () => {
  it('debería responder con un estado 200 en la ruta raíz', async () => {
    const response = await request(app).get('/');
    expect(response.status).toBe(200);
  });

  it('debería responder con un estado 200 y json en la ruta /api/health', async () => {
    const response = await request(app).get('/api/health');
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('status', 'ok');
  });
});
