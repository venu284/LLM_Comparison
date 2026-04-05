const { test, expect } = require('@playwright/test');
const { solutionUrl } = require('./helpers');

const pageUrl = solutionUrl('CSS-012.html');

const contentHeight = async (locator) => {
  const box = await locator.boundingBox();
  return box ? box.height : 0;
};

test.describe('CSS-012: CSS-Only Accordion', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(pageUrl);
  });

  test('content is hidden initially', async ({ page }) => {
    const firstHeight = await contentHeight(page.locator('.content').nth(0));
    const secondHeight = await contentHeight(page.locator('.content').nth(1));
    expect(firstHeight).toBeLessThanOrEqual(2);
    expect(secondHeight).toBeLessThanOrEqual(2);
  });

  test('clicking label 1 shows content 1', async ({ page }) => {
    await page.locator('label[for="section-1"]').click();
    await page.waitForTimeout(100);
    expect(await contentHeight(page.locator('.content').nth(0))).toBeGreaterThan(10);
  });

  test('clicking label 2 hides content 1', async ({ page }) => {
    await page.locator('label[for="section-1"]').click();
    await page.waitForTimeout(100);
    await page.locator('label[for="section-2"]').click();
    await page.waitForTimeout(100);
    expect(await contentHeight(page.locator('.content').nth(0))).toBeLessThanOrEqual(2);
  });

  test('clicking label 2 shows content 2', async ({ page }) => {
    await page.locator('label[for="section-2"]').click();
    await page.waitForTimeout(100);
    expect(await contentHeight(page.locator('.content').nth(1))).toBeGreaterThan(10);
  });

  test('only one section is open at a time', async ({ page }) => {
    await page.locator('label[for="section-1"]').click();
    await page.waitForTimeout(100);
    await page.locator('label[for="section-2"]').click();
    await page.waitForTimeout(100);
    const checkedCount = await page.locator('input[type="radio"]:checked').count();
    expect(checkedCount).toBe(1);
  });

  test('content panels use a transition property', async ({ page }) => {
    const transition = await page.locator('.content').first().evaluate((element) => getComputedStyle(element).transitionProperty);
    expect(transition).toContain('max-height');
  });

  test('radio inputs are present but hidden', async ({ page }) => {
    await expect(page.locator('input[type="radio"]')).toHaveCount(3);
    const opacity = await page.locator('input[type="radio"]').first().evaluate((element) => getComputedStyle(element).opacity);
    expect(opacity).toBe('0');
  });

  test('active label becomes bold with a different background color', async ({ page }) => {
    const defaultBackground = await page.locator('label[for="section-1"]').evaluate((element) => getComputedStyle(element).backgroundColor);
    await page.locator('label[for="section-1"]').click();
    await page.waitForTimeout(100);
    const activeBackground = await page.locator('label[for="section-1"]').evaluate((element) => getComputedStyle(element).backgroundColor);
    const fontWeight = await page.locator('label[for="section-1"]').evaluate((element) => getComputedStyle(element).fontWeight);
    expect(activeBackground).not.toBe(defaultBackground);
    expect(Number(fontWeight)).toBeGreaterThanOrEqual(700);
  });
});

