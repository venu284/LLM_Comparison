const { readSolution } = require('./helpers');

describe('CSS-007: Navigation with Dropdown', () => {
  const css = readSolution('CSS-007.css');

  test('uses the specified nav background', () => {
    expect(css).toMatch(/\.nav\s*{[^}]*background\s*:\s*#2c3e50/i);
  });

  test('nav text is white', () => {
    expect(css).toMatch(/\.nav\s*{[^}]*color\s*:\s*#ffffff/i);
  });

  test('dropdown is hidden by default', () => {
    expect(css).toMatch(/\.dropdown\s*{[^}]*opacity\s*:\s*0/i);
    expect(css).toMatch(/\.dropdown\s*{[^}]*visibility\s*:\s*hidden/i);
  });

  test('dropdown is absolutely positioned', () => {
    expect(css).toMatch(/\.dropdown\s*{[^}]*position\s*:\s*absolute/i);
  });

  test('dropdown has border and shadow styling', () => {
    expect(css).toMatch(/\.dropdown\s*{[^}]*border\s*:\s*1px/i);
    expect(css).toMatch(/\.dropdown\s*{[^}]*box-shadow/i);
  });

  test('dropdown appears on nav item hover', () => {
    expect(css).toMatch(/\.nav-item:hover\s+\.dropdown\s*{[^}]*opacity\s*:\s*1/i);
  });

  test('dropdown uses a transition', () => {
    expect(css).toMatch(/transition\s*:[^;]*150ms/i);
  });

  test('dropdown items have a hover style', () => {
    expect(css).toMatch(/\.dropdown-item:hover\s*{[^}]*background/i);
  });
});

