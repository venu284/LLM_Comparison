const request = require('supertest');
const { loadBugfixModule } = require('./helpers');

const app = loadBugfixModule('BF-003');

describe('BF-003: Off-by-One Pagination', () => {
  test('page 1 returns the first three items', async () => {
    const response = await request(app).get('/items').query({ page: 1, limit: 3 });
    expect(response.body.map((item) => item.id)).toEqual([1, 2, 3]);
  });

  test('page 2 returns the next three items', async () => {
    const response = await request(app).get('/items').query({ page: 2, limit: 3 });
    expect(response.body.map((item) => item.id)).toEqual([4, 5, 6]);
  });

  test('last page returns the remaining items without padding', async () => {
    const response = await request(app).get('/items').query({ page: 4, limit: 3 });
    expect(response.body.map((item) => item.id)).toEqual([10]);
  });

  test('page beyond the available data returns an empty array', async () => {
    const response = await request(app).get('/items').query({ page: 5, limit: 3 });
    expect(response.body).toEqual([]);
  });

  test('page 1 with the full limit returns every item', async () => {
    const response = await request(app).get('/items').query({ page: 1, limit: 10 });
    expect(response.body.map((item) => item.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });
});
