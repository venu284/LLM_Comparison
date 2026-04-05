const { readBugfixSource } = require('./helpers');

describe('BF-004: CSS Flexbox Direction', () => {
  const css = readBugfixSource('BF-004');

  test('mobile media query uses column direction', () => {
    expect(css).toMatch(/@media[\s\S]*\.layout\s*{[^}]*flex-direction\s*:\s*column/i);
  });

  test('desktop layout keeps row direction', () => {
    expect(css).toMatch(/\.layout\s*{[^}]*flex-direction\s*:\s*row/i);
  });
});
