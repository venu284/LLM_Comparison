const { readSolution } = require('./helpers');

describe('CSS-005: Card Grid with Hover Effects', () => {
  const css = readSolution('CSS-005.css');

  test('uses CSS Grid on the container', () => {
    expect(css).toMatch(/\.card-grid\s*{[^}]*display\s*:\s*grid/i);
  });

  test('defines grid-template-columns', () => {
    expect(css).toMatch(/grid-template-columns/i);
  });

  test('uses a 16px gap', () => {
    expect(css).toMatch(/gap\s*:\s*16px/i);
  });

  test('styles cards with white background and 8px radius', () => {
    expect(css).toMatch(/\.card\s*{[^}]*background\s*:\s*#ffffff/i);
    expect(css).toMatch(/\.card\s*{[^}]*border-radius\s*:\s*8px/i);
  });

  test('adds a transition for hover effects', () => {
    expect(css).toMatch(/transition\s*:[^;]*200ms/i);
  });

  test('hover state lifts the card', () => {
    expect(css).toMatch(/\.card:hover\s*{[^}]*transform\s*:\s*translateY\(-4px\)/i);
  });

  test('hover state adds the requested shadow', () => {
    expect(css).toMatch(/\.card:hover\s*{[^}]*box-shadow\s*:\s*0\s+8px\s+16px\s+rgba\(0,\s*0,\s*0,\s*0\.1\)/i);
  });

  test('includes tablet and desktop media queries', () => {
    expect(css).toMatch(/@media\s*\(min-width\s*:\s*768px\)/i);
    expect(css).toMatch(/@media\s*\(min-width\s*:\s*1024px\)/i);
  });
});

