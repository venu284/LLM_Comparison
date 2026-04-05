const { readSolution } = require('./helpers');

describe('CSS-006: Animated Loading Spinner', () => {
  const css = readSolution('CSS-006.css');

  test('defines keyframes spin', () => {
    expect(css).toMatch(/@keyframes\s+spin/i);
  });

  test('rotates to 360deg', () => {
    expect(css).toMatch(/rotate\(360deg\)/i);
  });

  test('spinner is circular', () => {
    expect(css).toMatch(/border-radius\s*:\s*50%/i);
  });

  test('spinner uses a blue border-top color', () => {
    expect(css).toMatch(/border-top-color\s*:\s*#3498db/i);
  });

  test('spinner animation is configured', () => {
    expect(css).toMatch(/animation\s*:\s*spin\s+1s\s+linear\s+infinite/i);
  });

  test('spinner container uses flexbox centering', () => {
    expect(css).toMatch(/\.spinner-container\s*{[^}]*display\s*:\s*flex/i);
    expect(css).toMatch(/\.spinner-container\s*{[^}]*justify-content\s*:\s*center/i);
  });

  test('spinner container aligns items to the center', () => {
    expect(css).toMatch(/\.spinner-container\s*{[^}]*align-items\s*:\s*center/i);
  });
});

