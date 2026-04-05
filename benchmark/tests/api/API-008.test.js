const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-008');
let app;

beforeEach(() => {
  delete require.cache[appPath];
  app = require(appPath);
});

describe('API-008: Rate Limiter Middleware', () => {
  const getWithIp = (ip) => request(app).get('/api/data').set('X-Forwarded-For', ip);

  test('first request returns 200', async () => {
    const response = await getWithIp('10.0.0.1');
    expect(response.status).toBe(200);
  });

  test('response includes X-RateLimit-Remaining header', async () => {
    const response = await getWithIp('10.0.0.1');
    expect(response.headers['x-ratelimit-remaining']).toBeDefined();
  });

  test('five requests succeed', async () => {
    for (let index = 0; index < 5; index += 1) {
      const response = await getWithIp('10.0.0.1');
      expect(response.status).toBe(200);
    }
  });

  test('sixth request returns 429', async () => {
    for (let index = 0; index < 5; index += 1) {
      await getWithIp('10.0.0.1');
    }
    const response = await getWithIp('10.0.0.1');
    expect(response.status).toBe(429);
  });

  test('error message is Too many requests', async () => {
    for (let index = 0; index < 5; index += 1) {
      await getWithIp('10.0.0.1');
    }
    const response = await getWithIp('10.0.0.1');
    expect(response.body.error).toBe('Too many requests');
  });

  test('response includes retryAfter when limited', async () => {
    for (let index = 0; index < 5; index += 1) {
      await getWithIp('10.0.0.1');
    }
    const response = await getWithIp('10.0.0.1');
    expect(response.body.retryAfter).toBeGreaterThan(0);
  });

  test('X-RateLimit-Remaining decreases after each request', async () => {
    const first = await getWithIp('10.0.0.1');
    const second = await getWithIp('10.0.0.1');
    expect(first.headers['x-ratelimit-remaining']).toBe('4');
    expect(second.headers['x-ratelimit-remaining']).toBe('3');
  });

  test('different IPs have separate limits', async () => {
    for (let index = 0; index < 5; index += 1) {
      await getWithIp('10.0.0.1');
    }
    const otherIpResponse = await getWithIp('10.0.0.2');
    expect(otherIpResponse.status).toBe(200);
    expect(otherIpResponse.headers['x-ratelimit-remaining']).toBe('4');
  });
});

