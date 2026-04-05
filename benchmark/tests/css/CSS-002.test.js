const { readSolution } = require('./helpers');

describe('CSS-002: Sticky Header', () => {
  const css = readSolution('CSS-002.css');

  test('uses sticky positioning', () => {
    expect(css).toMatch(/position\s*:\s*sticky/i);
  });

  test('pins the header to the top', () => {
    expect(css).toMatch(/top\s*:\s*0/i);
  });

  test('sets the header height to 60px', () => {
    expect(css).toMatch(/height\s*:\s*60px/i);
  });

  test('adds the requested box shadow', () => {
    expect(css).toMatch(/box-shadow\s*:\s*0\s+2px\s+4px\s+rgba\(0,\s*0,\s*0,\s*0\.1\)/i);
  });

  test('uses a positive z-index', () => {
    const match = css.match(/z-index\s*:\s*(\d+)/i);
    expect(match).not.toBeNull();
    expect(Number(match[1])).toBeGreaterThan(0);
  });
});

