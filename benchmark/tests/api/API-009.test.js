const request = require('supertest');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const appPath = require.resolve('../../solutions/api/API-009');
let app;

beforeEach(() => {
  jest.resetModules();
  delete process.env.JWT_SECRET;
  app = require(appPath);
});

describe('API-009: JWT Authentication System', () => {
  const registrationPayload = {
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    password: 'securepass123'
  };

  const registerUser = () => request(app).post('/auth/register').send(registrationPayload);

  test('POST /auth/register creates a user with status 201', async () => {
    const response = await registerUser();
    expect(response.status).toBe(201);
  });

  test('registered user response has no password field', async () => {
    const response = await registerUser();
    expect(response.body.password).toBeUndefined();
  });

  test('duplicate email returns 409', async () => {
    await registerUser();
    const response = await registerUser();
    expect(response.status).toBe(409);
  });

  test('POST /auth/login with valid credentials returns a token', async () => {
    await registerUser();
    const response = await request(app).post('/auth/login').send({
      email: registrationPayload.email,
      password: registrationPayload.password
    });
    expect(response.status).toBe(200);
    expect(response.body.token).toBeDefined();
  });

  test('POST /auth/login with wrong password returns 401', async () => {
    await registerUser();
    const response = await request(app).post('/auth/login').send({
      email: registrationPayload.email,
      password: 'wrong-password'
    });
    expect(response.status).toBe(401);
  });

  test('POST /auth/login with a non-existent email returns 401', async () => {
    const response = await request(app).post('/auth/login').send({
      email: 'missing@example.com',
      password: 'securepass123'
    });
    expect(response.status).toBe(401);
  });

  test('GET /auth/profile with a valid token returns the user', async () => {
    await registerUser();
    const loginResponse = await request(app).post('/auth/login').send({
      email: registrationPayload.email,
      password: registrationPayload.password
    });

    const profileResponse = await request(app)
      .get('/auth/profile')
      .set('Authorization', `Bearer ${loginResponse.body.token}`);

    expect(profileResponse.status).toBe(200);
    expect(profileResponse.body.email).toBe(registrationPayload.email);
  });

  test('GET /auth/profile without a token returns 401', async () => {
    const response = await request(app).get('/auth/profile');
    expect(response.status).toBe(401);
  });

  test('GET /auth/profile with an invalid token returns 401', async () => {
    const response = await request(app)
      .get('/auth/profile')
      .set('Authorization', 'Bearer not-a-real-token');
    expect(response.status).toBe(401);
  });

  test('token contains user id and email', async () => {
    await registerUser();
    const loginResponse = await request(app).post('/auth/login').send({
      email: registrationPayload.email,
      password: registrationPayload.password
    });

    const payload = jwt.verify(loginResponse.body.token, 'test-secret');
    expect(payload.id).toBe(1);
    expect(payload.email).toBe(registrationPayload.email);
  });

  test('register without required fields returns 400', async () => {
    const response = await request(app).post('/auth/register').send({ name: 'Ada' });
    expect(response.status).toBe(400);
  });

  test('password is stored hashed, not in plain text', async () => {
    await registerUser();
    const storedUser = app.locals.users[0];
    expect(storedUser.password).not.toBe(registrationPayload.password);
    expect(bcrypt.compareSync(registrationPayload.password, storedUser.password)).toBe(true);
  });

  test('registered user can immediately log in', async () => {
    await registerUser();
    const response = await request(app).post('/auth/login').send({
      email: registrationPayload.email,
      password: registrationPayload.password
    });
    expect(response.status).toBe(200);
  });
});
