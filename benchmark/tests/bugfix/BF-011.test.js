const request = require('supertest');
const { loadBugfixModule } = require('./helpers');

const app = loadBugfixModule('BF-011');

describe('BF-011: Express Error Swallowing', () => {
  const requestData = () => request(app).get('/data').timeout({ deadline: 500 });

  test('failed async operation returns 500', async () => {
    app.set('getData', async () => {
      throw new Error('Database unavailable');
    });

    const response = await requestData();
    expect(response.status).toBe(500);
  });

  test('error message is included in the response', async () => {
    app.set('getData', async () => {
      throw new Error('Database unavailable');
    });

    const response = await requestData();
    expect(response.body.error).toBe('Database unavailable');
  });

  test('server still responds after an async failure', async () => {
    app.set('getData', async () => {
      throw new Error('Database unavailable');
    });

    await requestData();

    const healthResponse = await request(app).get('/health');
    expect(healthResponse.status).toBe(200);
    expect(healthResponse.body.status).toBe('ok');
  });
});
