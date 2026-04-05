const { readSolution } = require('./helpers');

describe('CSS-004: Badge and Pill Styles', () => {
  const css = readSolution('CSS-004.css');

  test('badge is circular', () => {
    expect(css).toMatch(/\.badge\s*{[^}]*border-radius\s*:\s*50%/i);
  });

  test('badge uses the required 20px size', () => {
    expect(css).toMatch(/\.badge\s*{[^}]*width\s*:\s*20px/i);
    expect(css).toMatch(/\.badge\s*{[^}]*height\s*:\s*20px/i);
  });

  test('pill uses a 9999px border radius', () => {
    expect(css).toMatch(/\.pill\s*{[^}]*border-radius\s*:\s*9999px/i);
  });

  test('pill uses the requested padding and font size', () => {
    expect(css).toMatch(/padding\s*:\s*4px\s+12px/i);
    expect(css).toMatch(/font-size\s*:\s*14px/i);
  });

  test('success and warning variants exist', () => {
    expect(css).toMatch(/\.pill--success\s*{[^}]*background/i);
    expect(css).toMatch(/\.pill--warning\s*{[^}]*background/i);
  });

  test('danger variant exists', () => {
    expect(css).toMatch(/\.pill--danger\s*{[^}]*background/i);
  });
});

