const express = require('express');

const app = express();
const webhooks = [];
let nextId = 1;

app.use(express.json());
app.set('processWebhook', async () => undefined);

const findWebhook = (id) => webhooks.find((webhook) => webhook.id === id);

const processRecord = async (record) => {
  const processor = app.get('processWebhook');
  await Promise.resolve(processor(record.payload));
};

app.post('/webhooks', (req, res) => {
  const webhook = {
    id: String(nextId),
    payload: req.body,
    status: 'pending',
    retryCount: 0,
    timestamp: new Date().toISOString()
  };

  webhooks.push(webhook);
  nextId += 1;

  res.status(201).json(webhook);
});

app.get('/webhooks', (req, res) => {
  res.status(200).json(webhooks);
});

app.get('/webhooks/:id', (req, res) => {
  const webhook = findWebhook(req.params.id);

  if (!webhook) {
    return res.status(404).json({ error: 'Webhook not found' });
  }

  return res.status(200).json(webhook);
});

app.post('/webhooks/:id/process', async (req, res) => {
  const webhook = findWebhook(req.params.id);

  if (!webhook) {
    return res.status(404).json({ error: 'Webhook not found' });
  }

  try {
    await processRecord(webhook);
    webhook.status = 'processed';
    return res.status(200).json(webhook);
  } catch (error) {
    webhook.retryCount += 1;
    webhook.status = 'failed';
    return res.status(200).json(webhook);
  }
});

app.post('/webhooks/:id/retry', async (req, res) => {
  const webhook = findWebhook(req.params.id);

  if (!webhook) {
    return res.status(404).json({ error: 'Webhook not found' });
  }

  if (webhook.status !== 'failed') {
    return res.status(400).json({ error: 'Only failed webhooks can be retried' });
  }

  try {
    await processRecord(webhook);
    webhook.status = 'processed';
    return res.status(200).json(webhook);
  } catch (error) {
    webhook.retryCount += 1;
    webhook.status = webhook.retryCount > 3 ? 'dead' : 'failed';
    return res.status(200).json(webhook);
  }
});

module.exports = app;

