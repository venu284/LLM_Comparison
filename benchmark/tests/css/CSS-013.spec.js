const { test, expect } = require('@playwright/test');
const { solutionUrl } = require('./helpers');

const pageUrl = solutionUrl('CSS-013.html');

test.describe('CSS-013: Dark Mode Theme Toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(pageUrl);
  });

  test('light mode colors are applied by default', async ({ page }) => {
    const bodyBackground = await page.locator('body').evaluate((element) => getComputedStyle(element).backgroundColor);
    const cardBackground = await page.locator('.card').evaluate((element) => getComputedStyle(element).backgroundColor);
    expect(bodyBackground).toBe('rgb(255, 255, 255)');
    expect(cardBackground).toBe('rgb(245, 245, 245)');
  });

  test('theme toggle checkbox exists', async ({ page }) => {
    await expect(page.locator('#theme-toggle')).toHaveCount(1);
  });

  test('clicking the toggle sets data-theme to dark', async ({ page }) => {
    await page.locator('#theme-toggle').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });

  test('dark mode changes the body background', async ({ page }) => {
    await page.locator('#theme-toggle').click();
    const bodyBackground = await page.locator('body').evaluate((element) => getComputedStyle(element).backgroundColor);
    expect(bodyBackground).toBe('rgb(26, 26, 46)');
  });

  test('dark mode changes the card background', async ({ page }) => {
    await page.locator('#theme-toggle').click();
    const cardBackground = await page.locator('.card').evaluate((element) => getComputedStyle(element).backgroundColor);
    expect(cardBackground).toBe('rgb(22, 33, 62)');
  });

  test('dark mode updates the primary heading color', async ({ page }) => {
    await page.locator('#theme-toggle').click();
    const headingColor = await page.locator('.title').evaluate((element) => getComputedStyle(element).color);
    expect(headingColor).toBe('rgb(100, 181, 246)');
  });

  test('style sheet defines light and dark custom property blocks', async ({ page }) => {
    const styleText = await page.locator('style').innerText();
    expect(styleText).toContain(':root');
    expect(styleText).toContain('[data-theme="dark"]');
  });

  test('colors outside the theme blocks come from CSS variables', async ({ page }) => {
    const styleText = await page.locator('style').innerText();
    const withoutThemeBlocks = styleText
      .replace(/:root\s*{[\s\S]*?}/, '')
      .replace(/\[data-theme="dark"\]\s*{[\s\S]*?}/, '');

    expect(withoutThemeBlocks).not.toMatch(/#[0-9a-fA-F]{3,6}|rgba?\(/);
    expect(withoutThemeBlocks).toContain('var(--');
  });

  test('toggling again returns to light mode', async ({ page }) => {
    await page.locator('#theme-toggle').click();
    await page.locator('#theme-toggle').click();
    await expect(page.locator('html')).not.toHaveAttribute('data-theme', 'dark');
  });
});
