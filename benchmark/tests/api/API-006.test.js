const request = require('supertest');

const app = require('../../solutions/api/API-006');

describe('API-006: Paginated List with Sorting', () => {
  test('GET /products returns the first 5 products by default', async () => {
    const response = await request(app).get('/products');
    expect(response.body.data).toHaveLength(5);
    expect(response.body.data.map((item) => item.id)).toEqual([1, 2, 3, 4, 5]);
  });

  test('response has data array and pagination object', async () => {
    const response = await request(app).get('/products');
    expect(Array.isArray(response.body.data)).toBe(true);
    expect(response.body).toHaveProperty('pagination');
  });

  test('page=2 returns the next 5 products', async () => {
    const response = await request(app).get('/products').query({ page: 2 });
    expect(response.body.data.map((item) => item.id)).toEqual([6, 7, 8, 9, 10]);
  });

  test('limit=10 returns 10 products', async () => {
    const response = await request(app).get('/products').query({ limit: 10 });
    expect(response.body.data).toHaveLength(10);
  });

  test('sort=price&order=asc sorts by price ascending', async () => {
    const response = await request(app)
      .get('/products')
      .query({ sort: 'price', order: 'asc' });
    expect(response.body.data[0].price).toBe(10);
    expect(response.body.data[1].price).toBe(20);
  });

  test('sort=price&order=desc sorts by price descending', async () => {
    const response = await request(app)
      .get('/products')
      .query({ sort: 'price', order: 'desc' });
    expect(response.body.data[0].price).toBe(200);
    expect(response.body.data[1].price).toBe(190);
  });

  test('page beyond the available range returns an empty array', async () => {
    const response = await request(app).get('/products').query({ page: 99 });
    expect(response.body.data).toEqual([]);
  });

  test('pagination total is 20', async () => {
    const response = await request(app).get('/products');
    expect(response.body.pagination.total).toBe(20);
  });

  test('pagination totalPages is correct', async () => {
    const response = await request(app).get('/products').query({ limit: 6 });
    expect(response.body.pagination.totalPages).toBe(4);
  });

  test('default values are page 1, limit 5, sort id, order asc', async () => {
    const response = await request(app).get('/products');
    expect(response.body.pagination.page).toBe(1);
    expect(response.body.pagination.limit).toBe(5);
    expect(response.body.data[0].id).toBe(1);
  });
});

