const { readSolution } = require('./helpers');

describe('CSS-003: Responsive Two-Column Layout', () => {
  const css = readSolution('CSS-003.css');

  test('uses flexbox for the layout', () => {
    expect(css).toMatch(/\.layout\s*{[^}]*display\s*:\s*flex/i);
  });

  test('sets the sidebar width to 250px', () => {
    expect(css).toMatch(/\.sidebar\s*{[^}]*width\s*:\s*250px/i);
  });

  test('includes a mobile media query at 768px', () => {
    expect(css).toMatch(/@media\s*\(max-width\s*:\s*768px\)/i);
  });

  test('stacks columns vertically in the media query', () => {
    expect(css).toMatch(/@media[\s\S]*\.layout\s*{[^}]*flex-direction\s*:\s*column/i);
  });

  test('sets sidebar and main to full width on small screens', () => {
    expect(css).toMatch(/@media[\s\S]*width\s*:\s*100%/i);
  });
});

