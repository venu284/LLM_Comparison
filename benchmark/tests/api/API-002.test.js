const request = require('supertest');

const app = require('../../solutions/api/API-002');

describe('API-002: Echo Endpoint', () => {
  test('POST /echo with a body returns 200', async () => {
    const response = await request(app).post('/echo').send({ message: 'hello' });
    expect(response.status).toBe(200);
  });

  test('response wraps the body in received', async () => {
    const payload = { message: 'hello' };
    const response = await request(app).post('/echo').send(payload);
    expect(response.body).toEqual({ received: payload });
  });

  test('empty body returns 400', async () => {
    const response = await request(app).post('/echo').send({});
    expect(response.status).toBe(400);
  });

  test('error message matches', async () => {
    const response = await request(app).post('/echo').send({});
    expect(response.body.error).toBe('No body provided');
  });

  test('handles nested objects', async () => {
    const payload = { user: { name: 'Ada', role: 'admin' } };
    const response = await request(app).post('/echo').send(payload);
    expect(response.body.received).toEqual(payload);
  });

  test('content type is application/json', async () => {
    const response = await request(app).post('/echo').send({ ok: true });
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });
});

