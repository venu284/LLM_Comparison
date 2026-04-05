const request = require('supertest');

const app = require('../../solutions/api/API-001');

describe('API-001: Health Check Endpoint', () => {
  test('GET /health returns 200', async () => {
    const response = await request(app).get('/health');
    expect(response.status).toBe(200);
  });

  test('response has status equal to ok', async () => {
    const response = await request(app).get('/health');
    expect(response.body.status).toBe('ok');
  });

  test('response has a timestamp field', async () => {
    const response = await request(app).get('/health');
    expect(response.body).toHaveProperty('timestamp');
  });

  test('timestamp is valid ISO format', async () => {
    const response = await request(app).get('/health');
    expect(Number.isNaN(Date.parse(response.body.timestamp))).toBe(false);
  });

  test('content type is application/json', async () => {
    const response = await request(app).get('/health');
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });
});

