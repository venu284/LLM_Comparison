const { readSolution } = require('./helpers');

describe('CSS-001: Center a Card', () => {
  const css = readSolution('CSS-001.css');

  test('uses flexbox on .container', () => {
    expect(css).toMatch(/\.container\s*{[^}]*display\s*:\s*flex/i);
  });

  test('has justify-content center', () => {
    expect(css).toMatch(/justify-content\s*:\s*center/i);
  });

  test('has align-items center', () => {
    expect(css).toMatch(/align-items\s*:\s*center/i);
  });

  test('uses min-height 100vh on the container', () => {
    expect(css).toMatch(/min-height\s*:\s*100vh/i);
  });

  test('sets the card to 300px by 200px', () => {
    expect(css).toMatch(/\.card\s*{[^}]*width\s*:\s*300px/i);
    expect(css).toMatch(/\.card\s*{[^}]*height\s*:\s*200px/i);
  });
});

