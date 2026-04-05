const express = require('express');

const app = express();

app.set('getData', async () => ({ message: 'ok' }));

const asyncHandler = (handler) => (req, res, next) => {
  Promise.resolve(handler(req, res, next)).catch(next);
};

app.get(
  '/data',
  asyncHandler(async (req, res) => {
    const getData = app.get('getData');
    const data = await getData();
    res.status(200).json(data);
  })
);

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

app.use((error, req, res, next) => {
  res.status(500).json({
    error: error.message
  });
});

module.exports = app;

