const express = require('express');

const app = express();
const items = [];
let nextId = 1;

app.use(express.json());

app.get('/items', (req, res) => {
  res.status(200).json(items);
});

app.post('/items', (req, res) => {
  const { name } = req.body || {};

  if (!name) {
    return res.status(400).json({ error: 'Name is required' });
  }

  const item = {
    id: nextId,
    name
  };

  items.push(item);
  nextId += 1;

  return res.status(201).json(item);
});

module.exports = app;

