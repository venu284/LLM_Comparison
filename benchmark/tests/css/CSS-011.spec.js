const { test, expect } = require('@playwright/test');
const { solutionUrl } = require('./helpers');

const pageUrl = solutionUrl('CSS-011.html');

test.describe('CSS-011: Animated Card Flip', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(pageUrl);
  });

  test('flip card has the correct dimensions', async ({ page }) => {
    const box = await page.locator('.flip-card').boundingBox();
    expect(Math.round(box.width)).toBe(300);
    expect(Math.round(box.height)).toBe(200);
  });

  test('outer card uses 1000px perspective', async ({ page }) => {
    const perspective = await page.locator('.flip-card').evaluate((element) => getComputedStyle(element).perspective);
    expect(perspective).toBe('1000px');
  });

  test('inner card uses a 600ms transition', async ({ page }) => {
    const duration = await page
      .locator('.flip-card-inner')
      .evaluate((element) => getComputedStyle(element).transitionDuration);
    expect(duration).toBe('0.6s');
  });

  test('front side is visible initially', async ({ page }) => {
    const transform = await page
      .locator('.flip-card-inner')
      .evaluate((element) => getComputedStyle(element).transform);
    await expect(page.locator('.flip-card-front')).toContainText('Front Side');
    expect(transform).toBe('none');
  });

  test('back side contains the expected text', async ({ page }) => {
    await expect(page.locator('.flip-card-back')).toContainText('Back Side');
  });

  test('back face is configured with hidden backface visibility', async ({ page }) => {
    const visibility = await page
      .locator('.flip-card-back')
      .evaluate((element) => getComputedStyle(element).backfaceVisibility);
    expect(visibility).toBe('hidden');
  });

  test('hovering flips the inner card', async ({ page }) => {
    await page.locator('.flip-card').hover();
    await page.waitForTimeout(100);
    const transform = await page
      .locator('.flip-card-inner')
      .evaluate((element) => getComputedStyle(element).transform);
    expect(transform).not.toBe('none');
  });
});

