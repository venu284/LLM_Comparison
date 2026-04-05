const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-011');
let app;

beforeEach(() => {
  delete require.cache[appPath];
  app = require(appPath);
});

describe('API-011: Webhook Receiver with Retry Queue', () => {
  const createWebhook = async (payload = { event: 'build.finished' }) => {
    const response = await request(app).post('/webhooks').send(payload);
    return response.body;
  };

  test('POST /webhooks stores a webhook with pending status', async () => {
    const response = await request(app).post('/webhooks').send({ event: 'build.finished' });
    expect(response.status).toBe(201);
    expect(response.body.status).toBe('pending');
  });

  test('GET /webhooks lists all stored webhooks', async () => {
    await createWebhook();
    const response = await request(app).get('/webhooks');
    expect(response.body).toHaveLength(1);
  });

  test('GET /webhooks/:id returns a specific webhook', async () => {
    const webhook = await createWebhook();
    const response = await request(app).get(`/webhooks/${webhook.id}`);
    expect(response.body.id).toBe(webhook.id);
  });

  test('successful processing sets status to processed', async () => {
    app.set('processWebhook', jest.fn().mockResolvedValue(undefined));
    const webhook = await createWebhook();
    const response = await request(app).post(`/webhooks/${webhook.id}/process`);
    expect(response.body.status).toBe('processed');
  });

  test('failed processing sets status to failed', async () => {
    app.set('processWebhook', jest.fn().mockRejectedValue(new Error('boom')));
    const webhook = await createWebhook();
    const response = await request(app).post(`/webhooks/${webhook.id}/process`);
    expect(response.body.status).toBe('failed');
  });

  test('a failed webhook can be retried', async () => {
    app.set('processWebhook', jest.fn().mockRejectedValue(new Error('boom')));
    const webhook = await createWebhook();
    await request(app).post(`/webhooks/${webhook.id}/process`);
    app.set('processWebhook', jest.fn().mockResolvedValue(undefined));
    const retryResponse = await request(app).post(`/webhooks/${webhook.id}/retry`);
    expect(retryResponse.body.status).toBe('processed');
  });

  test('retry increments retryCount', async () => {
    app.set('processWebhook', jest.fn().mockRejectedValue(new Error('boom')));
    const webhook = await createWebhook();
    await request(app).post(`/webhooks/${webhook.id}/process`);
    const retryResponse = await request(app).post(`/webhooks/${webhook.id}/retry`);
    expect(retryResponse.body.retryCount).toBe(2);
  });

  test('after 3 retries the webhook becomes dead', async () => {
    app.set('processWebhook', jest.fn().mockRejectedValue(new Error('boom')));
    const webhook = await createWebhook();
    await request(app).post(`/webhooks/${webhook.id}/process`);
    await request(app).post(`/webhooks/${webhook.id}/retry`);
    await request(app).post(`/webhooks/${webhook.id}/retry`);
    const finalResponse = await request(app).post(`/webhooks/${webhook.id}/retry`);
    expect(finalResponse.body.status).toBe('dead');
  });

  test('cannot retry a webhook that is not failed', async () => {
    const webhook = await createWebhook();
    const response = await request(app).post(`/webhooks/${webhook.id}/retry`);
    expect(response.status).toBe(400);
  });

  test('cannot retry a dead webhook', async () => {
    app.set('processWebhook', jest.fn().mockRejectedValue(new Error('boom')));
    const webhook = await createWebhook();
    await request(app).post(`/webhooks/${webhook.id}/process`);
    await request(app).post(`/webhooks/${webhook.id}/retry`);
    await request(app).post(`/webhooks/${webhook.id}/retry`);
    await request(app).post(`/webhooks/${webhook.id}/retry`);
    const response = await request(app).post(`/webhooks/${webhook.id}/retry`);
    expect(response.status).toBe(400);
  });

  test('returns 404 for a non-existent webhook', async () => {
    const response = await request(app).get('/webhooks/999');
    expect(response.status).toBe(404);
  });
});

