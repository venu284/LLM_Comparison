const express = require('express');

const app = express();

app.use(express.json());

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const validate = (schema) => (req, res, next) => {
  const errors = [];

  Object.entries(schema).forEach(([field, rules]) => {
    const value = req.body ? req.body[field] : undefined;
    const isMissing =
      value === undefined || value === null || (typeof value === 'string' && value.trim() === '');

    if (rules.required && isMissing) {
      errors.push({
        field,
        message: `${field} is required`
      });
      return;
    }

    if (isMissing) {
      return;
    }

    if (rules.type === 'string' && typeof value !== 'string') {
      errors.push({
        field,
        message: `${field} must be a string`
      });
      return;
    }

    if (rules.type === 'number' && (typeof value !== 'number' || Number.isNaN(value))) {
      errors.push({
        field,
        message: `${field} must be a number`
      });
      return;
    }

    if (rules.type === 'email') {
      if (typeof value !== 'string' || !emailPattern.test(value)) {
        errors.push({
          field,
          message: `${field} must be a valid email`
        });
        return;
      }
    }

    if (rules.minLength !== undefined && typeof value === 'string' && value.length < rules.minLength) {
      errors.push({
        field,
        message: `${field} must be at least ${rules.minLength} characters`
      });
    }

    if (rules.min !== undefined && typeof value === 'number' && value < rules.min) {
      errors.push({
        field,
        message: `${field} must be greater than or equal to ${rules.min}`
      });
    }

    if (rules.max !== undefined && typeof value === 'number' && value > rules.max) {
      errors.push({
        field,
        message: `${field} must be less than or equal to ${rules.max}`
      });
    }
  });

  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  return next();
};

const registerSchema = {
  name: { required: true, type: 'string', minLength: 2 },
  email: { required: true, type: 'email' },
  age: { required: true, type: 'number', min: 18, max: 120 }
};

app.post('/register', validate(registerSchema), (req, res) => {
  res.status(200).json({
    message: 'Registered successfully'
  });
});

module.exports = app;

