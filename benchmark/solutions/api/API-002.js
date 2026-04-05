const express = require('express');

const app = express();

app.use(express.json());

app.post('/echo', (req, res) => {
  if (!req.body || Object.keys(req.body).length === 0) {
    return res.status(400).json({ error: 'No body provided' });
  }

  return res.status(200).json({ received: req.body });
});

module.exports = app;

