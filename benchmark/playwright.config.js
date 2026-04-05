const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/css',
  testMatch: '**/*.spec.js',
  use: {
    headless: true
  }
});

