const request = require('supertest');

const appPath = require.resolve('../../solutions/api/API-003');
let app;

beforeEach(() => {
  delete require.cache[appPath];
  app = require(appPath);
});

describe('API-003: Item CRUD Create and List', () => {
  test('GET /items returns an empty array initially', async () => {
    const response = await request(app).get('/items');
    expect(response.status).toBe(200);
    expect(response.body).toEqual([]);
  });

  test('POST /items with name returns 201', async () => {
    const response = await request(app).post('/items').send({ name: 'Notebook' });
    expect(response.status).toBe(201);
  });

  test('created item has id and name', async () => {
    const response = await request(app).post('/items').send({ name: 'Notebook' });
    expect(response.body).toEqual({ id: 1, name: 'Notebook' });
  });

  test('GET /items returns the added item', async () => {
    await request(app).post('/items').send({ name: 'Notebook' });
    const response = await request(app).get('/items');
    expect(response.body).toEqual([{ id: 1, name: 'Notebook' }]);
  });

  test('multiple POST requests increment ids correctly', async () => {
    await request(app).post('/items').send({ name: 'One' });
    const secondResponse = await request(app).post('/items').send({ name: 'Two' });
    expect(secondResponse.body.id).toBe(2);
  });

  test('POST /items without name returns 400', async () => {
    const response = await request(app).post('/items').send({});
    expect(response.status).toBe(400);
  });

  test('error message matches', async () => {
    const response = await request(app).post('/items').send({});
    expect(response.body.error).toBe('Name is required');
  });
});

