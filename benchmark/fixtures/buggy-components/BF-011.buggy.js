const express = require('express');

const app = express();

app.set('getData', async () => ({ message: 'ok' }));

app.get('/data', async (req, res) => {
  const getData = app.get('getData');
  const data = await getData();
  res.status(200).json(data);
});

module.exports = app;

