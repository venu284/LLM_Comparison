const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-013');
let app;

beforeEach(() => {
  jest.resetModules();
  app = require(appPath);
});

describe('API-013: Error Handling and Logging Middleware', () => {
  test('GET /test/not-found returns 404', async () => {
    const response = await request(app).get('/test/not-found');
    expect(response.status).toBe(404);
  });

  test('error response has the correct structure', async () => {
    const response = await request(app).get('/test/not-found');
    expect(response.body).toEqual({
      error: {
        message: 'Requested resource was not found',
        code: 'NOT_FOUND',
        status: 404
      }
    });
  });

  test('GET /test/not-found returns code NOT_FOUND', async () => {
    const response = await request(app).get('/test/not-found');
    expect(response.body.error.code).toBe('NOT_FOUND');
  });

  test('GET /test/validation returns 400 with VALIDATION_ERROR', async () => {
    const response = await request(app).get('/test/validation');
    expect(response.status).toBe(400);
    expect(response.body.error.code).toBe('VALIDATION_ERROR');
  });

  test('GET /test/unauthorized returns 401 with UNAUTHORIZED', async () => {
    const response = await request(app).get('/test/unauthorized');
    expect(response.status).toBe(401);
    expect(response.body.error.code).toBe('UNAUTHORIZED');
  });

  test('GET /test/unknown returns 500 with INTERNAL_ERROR', async () => {
    const response = await request(app).get('/test/unknown');
    expect(response.status).toBe(500);
    expect(response.body.error.code).toBe('INTERNAL_ERROR');
  });

  test('GET /logs returns the request log array', async () => {
    const response = await request(app).get('/logs');
    expect(Array.isArray(response.body)).toBe(true);
  });

  test('log entries have method, path, and timestamp', async () => {
    await request(app).get('/test/not-found');
    const response = await request(app).get('/logs');
    const entry = response.body.find((item) => item.path === '/test/not-found');
    expect(entry).toHaveProperty('method', 'GET');
    expect(entry).toHaveProperty('path', '/test/not-found');
    expect(Number.isNaN(Date.parse(entry.timestamp))).toBe(false);
  });

  test('multiple requests produce multiple log entries', async () => {
    await request(app).get('/test/validation');
    await request(app).get('/test/unauthorized');
    const response = await request(app).get('/logs');
    expect(response.body).toHaveLength(3);
  });

  test('error handler catches async errors', async () => {
    const response = await request(app).get('/test/unknown');
    expect(response.body.error.message).toBe('Unexpected failure');
  });

  test('non-existent routes return 404', async () => {
    const response = await request(app).get('/missing-route');
    expect(response.status).toBe(404);
  });

  test('error responses are always JSON', async () => {
    const response = await request(app).get('/missing-route');
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });
});
