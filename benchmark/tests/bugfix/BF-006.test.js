const request = require('supertest');
const { loadBugfixModule } = require('./helpers');

const app = loadBugfixModule('BF-006');

describe('BF-006: Express Middleware Order', () => {
  test('protected route returns 401 without a token', async () => {
    const response = await request(app).get('/protected');
    expect(response.status).toBe(401);
  });

  test('protected route returns 200 with a valid token', async () => {
    const response = await request(app)
      .get('/protected')
      .set('Authorization', 'Bearer test-token');

    expect(response.status).toBe(200);
    expect(response.body.message).toBe('Protected data');
  });

  test('wrong bearer token still returns 401', async () => {
    const response = await request(app)
      .get('/protected')
      .set('Authorization', 'Bearer wrong-token');

    expect(response.status).toBe(401);
  });

  test('malformed authorization header returns 401', async () => {
    const response = await request(app)
      .get('/protected')
      .set('Authorization', 'Token test-token');

    expect(response.status).toBe(401);
  });

  test('unauthorized response includes the error payload', async () => {
    const response = await request(app).get('/protected');
    expect(response.body).toEqual({ error: 'Unauthorized' });
  });
});
