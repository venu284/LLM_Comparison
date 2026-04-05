const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const app = express();
const users = [];
let nextId = 1;

const getJwtSecret = () => process.env.JWT_SECRET || 'test-secret';

app.use(express.json());
app.locals.users = users;

const sanitizeUser = (user) => ({
  id: user.id,
  name: user.name,
  email: user.email
});

app.post('/auth/register', (req, res) => {
  const { name, email, password } = req.body || {};

  if (!name || !email || !password) {
    return res.status(400).json({ error: 'Name, email, and password are required' });
  }

  if (users.some((user) => user.email === email)) {
    return res.status(409).json({ error: 'Email already exists' });
  }

  const user = {
    id: nextId,
    name,
    email,
    password: bcrypt.hashSync(password, 10)
  };

  users.push(user);
  nextId += 1;

  return res.status(201).json(sanitizeUser(user));
});

app.post('/auth/login', (req, res) => {
  const { email, password } = req.body || {};
  const user = users.find((candidate) => candidate.email === email);

  if (!user || !bcrypt.compareSync(password || '', user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const token = jwt.sign(
    { id: user.id, email: user.email },
    getJwtSecret(),
    { expiresIn: '1h' }
  );

  return res.status(200).json({ token });
});

const authenticate = (req, res, next) => {
  const authorization = req.headers.authorization || '';

  if (!authorization.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authorization token required' });
  }

  const token = authorization.slice('Bearer '.length);

  try {
    const payload = jwt.verify(token, getJwtSecret());
    const user = users.find((candidate) => candidate.id === payload.id);

    if (!user) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    req.user = user;
    return next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

app.get('/auth/profile', authenticate, (req, res) => {
  res.status(200).json(sanitizeUser(req.user));
});

module.exports = app;

