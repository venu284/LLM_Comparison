const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const STORE_FILE = path.join(__dirname, 'store.json');

app.use(express.json());

if (!fs.existsSync(STORE_FILE)) {
  fs.writeFileSync(STORE_FILE, JSON.stringify({}, null, 2));
}

let store;

try {
  store = JSON.parse(fs.readFileSync(STORE_FILE, 'utf8'));
} catch (error) {
  store = {};
  fs.writeFileSync(STORE_FILE, JSON.stringify(store, null, 2));
}

let writeQueue = Promise.resolve();

const persistStore = () => {
  writeQueue = writeQueue.then(() =>
    fs.promises.writeFile(STORE_FILE, JSON.stringify(store, null, 2))
  );
  return writeQueue;
};

app.locals.storeFile = STORE_FILE;

app.put('/store/:key', async (req, res) => {
  if (!req.body || !Object.prototype.hasOwnProperty.call(req.body, 'value')) {
    return res.status(400).json({ error: 'Value is required' });
  }

  store[req.params.key] = req.body.value;
  await persistStore();

  return res.status(200).json({
    key: req.params.key,
    value: store[req.params.key]
  });
});

app.get('/store/:key', (req, res) => {
  if (!Object.prototype.hasOwnProperty.call(store, req.params.key)) {
    return res.status(404).json({ error: 'Key not found' });
  }

  return res.status(200).json({
    key: req.params.key,
    value: store[req.params.key]
  });
});

app.delete('/store/:key', async (req, res) => {
  if (!Object.prototype.hasOwnProperty.call(store, req.params.key)) {
    return res.status(404).json({ error: 'Key not found' });
  }

  delete store[req.params.key];
  await persistStore();
  return res.status(204).send();
});

app.get('/store', (req, res) => {
  res.status(200).json(store);
});

module.exports = app;

