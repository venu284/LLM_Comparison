const { readBugfixSource } = require('./helpers');

describe('BF-004: CSS Flexbox Direction', () => {
  const css = readBugfixSource('BF-004');
  const baseRuleMatch = css.match(/\.layout\s*{([^}]*)}/i);
  const mobileRuleMatch = css.match(/@media[\s\S]*?\.layout\s*{([^}]*)}/i);
  const baseRule = baseRuleMatch ? baseRuleMatch[1] : '';
  const mobileRule = mobileRuleMatch ? mobileRuleMatch[1] : '';

  test('mobile media query uses column direction', () => {
    expect(mobileRule).toMatch(/flex-direction\s*:\s*column/i);
  });

  test('desktop layout keeps row direction', () => {
    expect(baseRule).toMatch(/flex-direction\s*:\s*row/i);
  });

  test('base layout rule keeps display flex enabled', () => {
    expect(baseRule).toMatch(/display\s*:\s*flex/i);
  });

  test('mobile media query no longer keeps row direction', () => {
    expect(mobileRule).not.toMatch(/flex-direction\s*:\s*row/i);
  });

  test('base layout rule does not switch to column direction', () => {
    expect(baseRule).not.toMatch(/flex-direction\s*:\s*column/i);
  });
});
