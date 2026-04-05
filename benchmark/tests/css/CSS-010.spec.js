const { test, expect } = require('@playwright/test');
const { solutionUrl } = require('./helpers');

const pageUrl = solutionUrl('CSS-010.html');

const openPage = async (page, width, height) => {
  await page.setViewportSize({ width, height });
  await page.goto(pageUrl);
};

const columnCount = async (page) =>
  page.locator('.grid-container').evaluate((element) => {
    const template = getComputedStyle(element).gridTemplateColumns;
    return template.split(' ').filter(Boolean).length;
  });

test.describe('CSS-010: Responsive 3-Column Grid', () => {
  test('renders 9 grid items', async ({ page }) => {
    await openPage(page, 1024, 768);
    await expect(page.locator('.grid-item')).toHaveCount(9);
  });

  test('uses 3 columns on desktop', async ({ page }) => {
    await openPage(page, 1024, 768);
    expect(await columnCount(page)).toBe(3);
  });

  test('uses 2 columns on tablet', async ({ page }) => {
    await openPage(page, 600, 800);
    expect(await columnCount(page)).toBe(2);
  });

  test('uses 1 column on mobile', async ({ page }) => {
    await openPage(page, 375, 667);
    expect(await columnCount(page)).toBe(1);
  });

  test('uses a 16px gap', async ({ page }) => {
    await openPage(page, 1024, 768);
    const gap = await page.locator('.grid-container').evaluate((element) => getComputedStyle(element).gap);
    expect(gap).toBe('16px');
  });

  test('grid items use 16px padding', async ({ page }) => {
    await openPage(page, 1024, 768);
    const padding = await page.locator('.grid-item').first().evaluate((element) => getComputedStyle(element).padding);
    expect(padding).toBe('16px');
  });

  test('grid items use 8px border radius', async ({ page }) => {
    await openPage(page, 1024, 768);
    const radius = await page.locator('.grid-item').first().evaluate((element) => getComputedStyle(element).borderRadius);
    expect(radius).toBe('8px');
  });

  test('grid uses equal auto rows', async ({ page }) => {
    await openPage(page, 1024, 768);
    const autoRows = await page
      .locator('.grid-container')
      .evaluate((element) => getComputedStyle(element).gridAutoRows);
    expect(autoRows).toBe('1fr');
  });
});

