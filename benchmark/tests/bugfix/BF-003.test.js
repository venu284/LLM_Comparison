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
});
