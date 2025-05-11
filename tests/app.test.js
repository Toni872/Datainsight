import request from 'supertest';
import app from '../src/app'; // Asegúrate de que la ruta sea correcta

describe('Pruebas de la aplicación', () => {
    it('debería responder con un estado 200 en la ruta raíz', async () => {
        const response = await request(app).get('/');
        expect(response.status).toBe(200);
    });

    // Agrega más pruebas según sea necesario
});