const request = require('supertest');

const app = require('../../solutions/api/API-004');

describe('API-004: Query Parameter Greeting', () => {
  test('GET /greet?name=Alice returns Hello, Alice!', async () => {
    const response = await request(app).get('/greet').query({ name: 'Alice' });
    expect(response.body.message).toBe('Hello, Alice!');
  });

  test('GET /greet without name returns Hello, World!', async () => {
    const response = await request(app).get('/greet');
    expect(response.body.message).toBe('Hello, World!');
  });

  test('GET /greet with uppercase=true uppercases the message', async () => {
    const response = await request(app)
      .get('/greet')
      .query({ name: 'Bob', uppercase: 'true' });
    expect(response.body.message).toBe('HELLO, BOB!');
  });

  test('response is JSON', async () => {
    const response = await request(app).get('/greet');
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });

  test('status is 200', async () => {
    const response = await request(app).get('/greet');
    expect(response.status).toBe(200);
  });

  test('uppercase=true works with the default name', async () => {
    const response = await request(app).get('/greet').query({ uppercase: 'true' });
    expect(response.body.message).toBe('HELLO, WORLD!');
  });
});

