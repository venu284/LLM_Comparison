const express = require('express');

const app = express();
const logs = [];

class AppError extends Error {
  constructor(message, status, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

class NotFoundError extends AppError {
  constructor(message = 'Not found') {
    super(message, 404, 'NOT_FOUND');
  }
}

class ValidationError extends AppError {
  constructor(message = 'Validation failed') {
    super(message, 400, 'VALIDATION_ERROR');
  }
}

class UnauthorizedError extends AppError {
  constructor(message = 'Unauthorized') {
    super(message, 401, 'UNAUTHORIZED');
  }
}

const asyncHandler = (handler) => (req, res, next) => {
  Promise.resolve(handler(req, res, next)).catch(next);
};

app.locals.logs = logs;

app.use((req, res, next) => {
  logs.push({
    method: req.method,
    path: req.path,
    timestamp: new Date().toISOString()
  });
  next();
});

app.get('/logs', (req, res) => {
  res.status(200).json(logs);
});

app.get('/test/not-found', (req, res, next) => {
  next(new NotFoundError('Requested resource was not found'));
});

app.get('/test/validation', (req, res, next) => {
  next(new ValidationError('Validation failed'));
});

app.get('/test/unauthorized', (req, res, next) => {
  next(new UnauthorizedError('Unauthorized'));
});

app.get(
  '/test/unknown',
  asyncHandler(async () => {
    throw new Error('Unexpected failure');
  })
);

app.use((req, res, next) => {
  next(new NotFoundError('Route not found'));
});

app.use((error, req, res, next) => {
  const status = error.status || 500;
  const code = error.code || 'INTERNAL_ERROR';

  res.status(status).json({
    error: {
      message: error.message || 'Internal server error',
      code,
      status
    }
  });
});

module.exports = app;
