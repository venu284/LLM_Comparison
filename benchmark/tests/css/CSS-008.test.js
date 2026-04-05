const { readSolution } = require('./helpers');

describe('CSS-008: Form Styling', () => {
  const css = readSolution('CSS-008.css');

  test('form labels are block-level and bold', () => {
    expect(css).toMatch(/\.form-label\s*{[^}]*display\s*:\s*block/i);
    expect(css).toMatch(/\.form-label\s*{[^}]*font-weight\s*:\s*700/i);
  });

  test('inputs use the required height and padding', () => {
    expect(css).toMatch(/\.form-input\s*{[^}]*height\s*:\s*40px/i);
    expect(css).toMatch(/\.form-input\s*{[^}]*padding\s*:\s*12px/i);
  });

  test('inputs use the required border and radius', () => {
    expect(css).toMatch(/border\s*:\s*1px\s+solid\s+#ddd/i);
    expect(css).toMatch(/border-radius\s*:\s*4px/i);
  });

  test('focus state changes the border color', () => {
    expect(css).toMatch(/\.form-input:focus\s*{[^}]*border-color\s*:\s*#3498db/i);
  });

  test('focus state adds the blue box shadow', () => {
    expect(css).toMatch(/\.form-input:focus\s*{[^}]*box-shadow\s*:\s*0\s+0\s+0\s+3px\s+rgba\(52,\s*152,\s*219,\s*0\.1\)/i);
  });

  test('focus state removes the default outline', () => {
    expect(css).toMatch(/outline\s*:\s*none/i);
  });

  test('invalid inputs use red border styling', () => {
    expect(css).toMatch(/\.form-input\.invalid\s*{[^}]*border-color\s*:\s*#ef4444/i);
    expect(css).toMatch(/\.form-input\.invalid\s*{[^}]*box-shadow/i);
  });

  test('submit button uses the required base and hover styles', () => {
    expect(css).toMatch(/\.form-button\s*{[^}]*height\s*:\s*44px/i);
    expect(css).toMatch(/\.form-button\s*{[^}]*background\s*:\s*#3498db/i);
    expect(css).toMatch(/\.form-button:hover\s*{[^}]*background/i);
  });
});

