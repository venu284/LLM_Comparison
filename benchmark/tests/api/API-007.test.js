const request = require('supertest');

const app = require('../../solutions/api/API-007');

describe('API-007: Request Validation Middleware', () => {
  const validPayload = {
    name: 'Ada',
    email: 'ada@example.com',
    age: 36
  };

  test('valid input returns 200', async () => {
    const response = await request(app).post('/register').send(validPayload);
    expect(response.status).toBe(200);
  });

  test('missing name returns 400 with an error', async () => {
    const response = await request(app)
      .post('/register')
      .send({ email: 'ada@example.com', age: 36 });
    expect(response.status).toBe(400);
    expect(response.body.errors.some((error) => error.field === 'name')).toBe(true);
  });

  test('short name returns 400 with a minLength error', async () => {
    const response = await request(app)
      .post('/register')
      .send({ ...validPayload, name: 'A' });
    expect(response.body.errors[0].message).toMatch(/at least 2 characters/);
  });

  test('invalid email returns 400', async () => {
    const response = await request(app)
      .post('/register')
      .send({ ...validPayload, email: 'invalid-email' });
    expect(response.status).toBe(400);
  });

  test('age below 18 returns 400', async () => {
    const response = await request(app)
      .post('/register')
      .send({ ...validPayload, age: 17 });
    expect(response.body.errors[0].message).toMatch(/greater than or equal to 18/);
  });

  test('age above 120 returns 400', async () => {
    const response = await request(app)
      .post('/register')
      .send({ ...validPayload, age: 121 });
    expect(response.body.errors[0].message).toMatch(/less than or equal to 120/);
  });

  test('non-number age returns 400', async () => {
    const response = await request(app)
      .post('/register')
      .send({ ...validPayload, age: 'old' });
    expect(response.body.errors[0].message).toMatch(/must be a number/);
  });

  test('multiple errors are returned at once', async () => {
    const response = await request(app)
      .post('/register')
      .send({ name: '', email: 'invalid-email', age: 10 });
    expect(response.body.errors.length).toBeGreaterThanOrEqual(3);
  });

  test('error response has an errors array', async () => {
    const response = await request(app).post('/register').send({});
    expect(Array.isArray(response.body.errors)).toBe(true);
  });

  test('each error has field and message', async () => {
    const response = await request(app).post('/register').send({});
    response.body.errors.forEach((error) => {
      expect(error).toHaveProperty('field');
      expect(error).toHaveProperty('message');
    });
  });
});

