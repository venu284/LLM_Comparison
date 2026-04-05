const express = require('express');

const app = express();

app.get('/greet', (req, res) => {
  const name = req.query.name || 'World';
  let message = `Hello, ${name}!`;

  if (req.query.uppercase === 'true') {
    message = message.toUpperCase();
  }

  res.status(200).json({ message });
});

module.exports = app;

