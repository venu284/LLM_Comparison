const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-012');
let app;

beforeEach(() => {
  delete require.cache[appPath];
  app = require(appPath);
});

describe('API-012: API Versioning with Router', () => {
  test('GET /api/v1/users returns users with name only', async () => {
    const response = await request(app).get('/api/v1/users');
    expect(response.body.users[0]).toEqual({ name: 'Ada Lovelace' });
  });

  test('GET /api/v2/users returns name, email, and createdAt', async () => {
    const response = await request(app).get('/api/v2/users');
    expect(response.body.users[0]).toEqual({
      name: 'Ada Lovelace',
      email: 'ada@example.com',
      createdAt: '2024-01-01T00:00:00.000Z'
    });
  });

  test('v2 response includes a meta object', async () => {
    const response = await request(app).get('/api/v2/users');
    expect(response.body.meta).toEqual({
      version: 'v2',
      count: 1
    });
  });

  test('POST /api/v2/users creates a user', async () => {
    const response = await request(app)
      .post('/api/v2/users')
      .send({ name: 'Grace Hopper', email: 'grace@example.com' });
    expect(response.status).toBe(201);
    expect(response.body.name).toBe('Grace Hopper');
  });

  test('created user is visible in both v1 and v2', async () => {
    await request(app)
      .post('/api/v2/users')
      .send({ name: 'Grace Hopper', email: 'grace@example.com' });

    const v1Response = await request(app).get('/api/v1/users');
    const v2Response = await request(app).get('/api/v2/users');

    expect(v1Response.body.users).toContainEqual({ name: 'Grace Hopper' });
    expect(
      v2Response.body.users.some((user) => user.email === 'grace@example.com')
    ).toBe(true);
  });

  test('v1 does not expose email', async () => {
    const response = await request(app).get('/api/v1/users');
    expect(response.body.users[0].email).toBeUndefined();
  });

  test('GET /api/version returns version information', async () => {
    const response = await request(app).get('/api/version');
    expect(response.body).toEqual({
      versions: ['v1', 'v2'],
      latest: 'v2'
    });
  });

  test('invalid version returns 404', async () => {
    const response = await request(app).get('/api/v3/users');
    expect(response.status).toBe(404);
  });

  test('POST to v1 returns 404', async () => {
    const response = await request(app)
      .post('/api/v1/users')
      .send({ name: 'Grace Hopper', email: 'grace@example.com' });
    expect(response.status).toBe(404);
  });

  test('v2 meta.count is correct', async () => {
    const response = await request(app).get('/api/v2/users');
    expect(response.body.meta.count).toBe(1);
  });
});

