const express = require('express');

const app = express();

const products = Array.from({ length: 20 }, (_, index) => ({
  id: index + 1,
  name: `Product ${index + 1}`,
  price: (index + 1) * 10,
  category: index % 2 === 0 ? 'hardware' : 'software'
}));

const allowedSortFields = new Set(['id', 'name', 'price', 'category']);

app.get('/products', (req, res) => {
  const page = Math.max(1, Number.parseInt(req.query.page, 10) || 1);
  const limit = Math.max(1, Number.parseInt(req.query.limit, 10) || 5);
  const sort = allowedSortFields.has(req.query.sort) ? req.query.sort : 'id';
  const order = req.query.order === 'desc' ? 'desc' : 'asc';

  const sortedProducts = [...products].sort((left, right) => {
    const leftValue = left[sort];
    const rightValue = right[sort];

    if (leftValue < rightValue) {
      return order === 'asc' ? -1 : 1;
    }

    if (leftValue > rightValue) {
      return order === 'asc' ? 1 : -1;
    }

    return 0;
  });

  const total = sortedProducts.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const data = sortedProducts.slice(startIndex, startIndex + limit);

  res.status(200).json({
    data,
    pagination: {
      page,
      limit,
      total,
      totalPages
    }
  });
});

module.exports = app;

