const request = require('supertest');
const { loadBugfixModule } = require('./helpers');

const app = loadBugfixModule('BF-011');
const defaultGetData = async () => ({ message: 'ok' });

describe('BF-011: Express Error Swallowing', () => {
  const requestData = () => request(app).get('/data').timeout({ deadline: 500 });

  beforeEach(() => {
    app.set('getData', defaultGetData);
  });

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

  test('successful async operations still return data', async () => {
    const response = await request(app).get('/data');
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: 'ok' });
  });

  test('error responses are returned as JSON', async () => {
    app.set('getData', async () => {
      throw new Error('Database unavailable');
    });

    const response = await requestData();
    expect(response.headers['content-type']).toMatch(/application\/json/);
  });
});
