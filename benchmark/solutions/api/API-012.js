const express = require('express');

const app = express();
const v1Router = express.Router();
const v2Router = express.Router();

const users = [
  {
    id: 1,
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    createdAt: '2024-01-01T00:00:00.000Z'
  }
];
let nextId = 2;

app.use(express.json());

v1Router.get('/users', (req, res) => {
  res.status(200).json({
    users: users.map((user) => ({ name: user.name }))
  });
});

v2Router.get('/users', (req, res) => {
  res.status(200).json({
    users: users.map((user) => ({
      name: user.name,
      email: user.email,
      createdAt: user.createdAt
    })),
    meta: {
      version: 'v2',
      count: users.length
    }
  });
});

v2Router.post('/users', (req, res) => {
  const { name, email } = req.body || {};

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required' });
  }

  const user = {
    id: nextId,
    name,
    email,
    createdAt: new Date().toISOString()
  };

  users.push(user);
  nextId += 1;

  return res.status(201).json(user);
});

app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

app.get('/api/version', (req, res) => {
  res.status(200).json({
    versions: ['v1', 'v2'],
    latest: 'v2'
  });
});

module.exports = app;

