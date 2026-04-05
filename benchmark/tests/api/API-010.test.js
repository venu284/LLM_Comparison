const fs = require('fs');
const path = require('path');
const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-010');
const storeFile = path.join(path.dirname(appPath), 'store.json');
let app;

beforeEach(() => {
  delete require.cache[appPath];
  if (fs.existsSync(storeFile)) {
    fs.unlinkSync(storeFile);
  }
  app = require(appPath);
});

afterAll(() => {
  if (fs.existsSync(storeFile)) {
    fs.unlinkSync(storeFile);
  }
});

describe('API-010: File-Based Key-Value Store API', () => {
  test('PUT /store/:key creates a key-value pair', async () => {
    const response = await request(app).put('/store/theme').send({ value: 'dark' });
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ key: 'theme', value: 'dark' });
  });

  test('GET /store/:key retrieves the stored value', async () => {
    await request(app).put('/store/theme').send({ value: 'dark' });
    const response = await request(app).get('/store/theme');
    expect(response.body.value).toBe('dark');
  });

  test('GET /store/:key for a missing key returns 404', async () => {
    const response = await request(app).get('/store/missing');
    expect(response.status).toBe(404);
  });

  test('DELETE /store/:key removes the key', async () => {
    await request(app).put('/store/theme').send({ value: 'dark' });
    const response = await request(app).delete('/store/theme');
    expect(response.status).toBe(204);
  });

  test('DELETE /store/:key for a missing key returns 404', async () => {
    const response = await request(app).delete('/store/missing');
    expect(response.status).toBe(404);
  });

  test('GET /store returns all key-value pairs', async () => {
    await request(app).put('/store/theme').send({ value: 'dark' });
    await request(app).put('/store/layout').send({ value: 'grid' });
    const response = await request(app).get('/store');
    expect(response.body).toEqual({ theme: 'dark', layout: 'grid' });
  });

  test('data persists to the JSON file', async () => {
    await request(app).put('/store/theme').send({ value: 'dark' });
    const storedData = JSON.parse(fs.readFileSync(app.locals.storeFile, 'utf8'));
    expect(storedData).toEqual({ theme: 'dark' });
  });

  test('handles string values', async () => {
    await request(app).put('/store/name').send({ value: 'benchmark' });
    const response = await request(app).get('/store/name');
    expect(response.body.value).toBe('benchmark');
  });

  test('handles object values', async () => {
    const value = { enabled: true, retries: 3 };
    await request(app).put('/store/config').send({ value });
    const response = await request(app).get('/store/config');
    expect(response.body.value).toEqual(value);
  });

  test('handles array values', async () => {
    const value = ['a', 'b', 'c'];
    await request(app).put('/store/list').send({ value });
    const response = await request(app).get('/store/list');
    expect(response.body.value).toEqual(value);
  });

  test('updating an existing key overwrites the value', async () => {
    await request(app).put('/store/theme').send({ value: 'dark' });
    await request(app).put('/store/theme').send({ value: 'light' });
    const response = await request(app).get('/store/theme');
    expect(response.body.value).toBe('light');
  });
});

