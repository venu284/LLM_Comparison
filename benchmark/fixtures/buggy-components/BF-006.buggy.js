const express = require('express');

const app = express();

const auth = (req, res, next) => {
  if (req.headers.authorization !== 'Bearer test-token') {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  return next();
};

app.get('/protected', (req, res) => {
  res.status(200).json({ message: 'Protected data' });
});

app.use(auth);

module.exports = app;

