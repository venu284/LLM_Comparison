const express = require('express');

const app = express();
const users = [];
let nextId = 1;

app.use(express.json());

const parseId = (value) => Number.parseInt(value, 10);

const findUser = (id) => users.find((user) => user.id === id);

app.get('/users', (req, res) => {
  res.status(200).json(users);
});

app.get('/users/:id', (req, res) => {
  const user = findUser(parseId(req.params.id));

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  return res.status(200).json(user);
});

app.post('/users', (req, res) => {
  const { name, email } = req.body || {};

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required' });
  }

  const user = {
    id: nextId,
    name,
    email
  };

  users.push(user);
  nextId += 1;

  return res.status(201).json(user);
});

app.put('/users/:id', (req, res) => {
  const user = findUser(parseId(req.params.id));

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  const { name, email } = req.body || {};

  if (name !== undefined) {
    user.name = name;
  }

  if (email !== undefined) {
    user.email = email;
  }

  return res.status(200).json(user);
});

app.delete('/users/:id', (req, res) => {
  const id = parseId(req.params.id);
  const index = users.findIndex((user) => user.id === id);

  if (index === -1) {
    return res.status(404).json({ error: 'User not found' });
  }

  users.splice(index, 1);
  return res.status(204).send();
});

module.exports = app;
