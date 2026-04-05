const express = require('express');

const app = express();

app.set('trust proxy', true);

const createRateLimiter = ({ max, windowMs }) => {
  const entries = new Map();

  return (req, res, next) => {
    const now = Date.now();
    const ip = req.ip || 'unknown';
    let entry = entries.get(ip);

    if (!entry || entry.resetTime <= now) {
      entry = {
        count: 0,
        resetTime: now + windowMs
      };
      entries.set(ip, entry);
    }

    if (entry.count >= max) {
      const retryAfter = Math.max(1, Math.ceil((entry.resetTime - now) / 1000));
      res.set('X-RateLimit-Remaining', '0');
      return res.status(429).json({
        error: 'Too many requests',
        retryAfter
      });
    }

    entry.count += 1;
    res.set('X-RateLimit-Remaining', String(Math.max(0, max - entry.count)));
    return next();
  };
};

app.use(createRateLimiter({ max: 5, windowMs: 60_000 }));

app.get('/api/data', (req, res) => {
  res.status(200).json({ message: 'Success' });
});

module.exports = app;
