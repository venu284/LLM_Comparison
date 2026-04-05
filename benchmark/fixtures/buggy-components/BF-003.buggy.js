const express = require('express');

const app = express();
const items = Array.from({ length: 10 }, (_, index) => ({
  id: index + 1,
  name: `Item ${index + 1}`
}));

app.get('/items', (req, res) => {
  const page = Number.parseInt(req.query.page, 10) || 1;
  const limit = Number.parseInt(req.query.limit, 10) || 3;
  const data = items.slice(page * limit, (page + 1) * limit);

  res.status(200).json(data);
});

module.exports = app;

