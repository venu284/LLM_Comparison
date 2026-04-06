const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-005');
let app;

beforeEach(() => {
  jest.resetModules();
  app = require(appPath);
});

describe('API-005: RESTful CRUD for Users', () => {
  const createUser = () =>
    request(app).post('/users').send({ name: 'Ada Lovelace', email: 'ada@example.com' });

  test('GET /users returns an empty array', async () => {
    const response = await request(app).get('/users');
    expect(response.body).toEqual([]);
  });

  test('POST /users creates a user and returns 201', async () => {
    const response = await createUser();
    expect(response.status).toBe(201);
  });

  test('created user has id, name, and email', async () => {
    const response = await createUser();
    expect(response.body).toEqual({
      id: 1,
      name: 'Ada Lovelace',
      email: 'ada@example.com'
    });
  });

  test('GET /users/:id returns a specific user', async () => {
    await createUser();
    const response = await request(app).get('/users/1');
    expect(response.body.name).toBe('Ada Lovelace');
  });

  test('GET /users/:id with an invalid id returns 404', async () => {
    const response = await request(app).get('/users/999');
    expect(response.status).toBe(404);
  });

  test('PUT /users/:id updates user fields', async () => {
    await createUser();
    const response = await request(app).put('/users/1').send({ name: 'Grace Hopper' });
    expect(response.body).toEqual({
      id: 1,
      name: 'Grace Hopper',
      email: 'ada@example.com'
    });
  });

  test('PUT /users/:id with an invalid id returns 404', async () => {
    const response = await request(app).put('/users/999').send({ name: 'Grace Hopper' });
    expect(response.status).toBe(404);
  });

  test('DELETE /users/:id removes the user and returns 204', async () => {
    await createUser();
    const response = await request(app).delete('/users/1');
    expect(response.status).toBe(204);
  });

  test('DELETE /users/:id with an invalid id returns 404', async () => {
    const response = await request(app).delete('/users/999');
    expect(response.status).toBe(404);
  });

  test('POST /users without name returns 400', async () => {
    const response = await request(app).post('/users').send({ email: 'ada@example.com' });
    expect(response.status).toBe(400);
  });

  test('POST /users without email returns 400', async () => {
    const response = await request(app).post('/users').send({ name: 'Ada Lovelace' });
    expect(response.status).toBe(400);
  });

  test('GET /users after delete no longer includes the deleted user', async () => {
    await createUser();
    await request(app).delete('/users/1');
    const response = await request(app).get('/users');
    expect(response.body).toEqual([]);
  });
});
