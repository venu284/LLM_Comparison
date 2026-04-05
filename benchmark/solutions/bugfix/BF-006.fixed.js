const express = require('express');

const app = express();

const auth = (req, res, next) => {
  if (req.headers.authorization !== 'Bearer test-token') {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  return next();
};

app.use('/protected', auth);

app.get('/protected', (req, res) => {
  res.status(200).json({ message: 'Protected data' });
});

module.exports = app;

