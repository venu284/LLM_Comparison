const { readSolution } = require('./helpers');

describe('CSS-009: Holy Grail Layout', () => {
  const css = readSolution('CSS-009.css');

  test('uses CSS Grid on the page container', () => {
    expect(css).toMatch(/\.page\s*{[^}]*display\s*:\s*grid/i);
  });

  test('defines grid areas or columns for the layout', () => {
    expect(css).toMatch(/grid-template-areas|grid-template-columns/i);
  });

  test('fills the viewport height', () => {
    expect(css).toMatch(/min-height\s*:\s*100vh/i);
  });

  test('uses 200px sidebars in the desktop layout', () => {
    expect(css).toMatch(/grid-template-columns\s*:\s*200px\s+minmax\(0,\s*1fr\)\s+200px/i);
  });

  test('sets header and footer row heights to 60px', () => {
    expect(css).toMatch(/grid-template-rows\s*:\s*60px\s+1fr\s+60px/i);
  });

  test('includes a breakpoint at 768px', () => {
    expect(css).toMatch(/@media\s*\(max-width\s*:\s*768px\)/i);
  });

  test('stacks the layout at the 768px breakpoint', () => {
    expect(css).toMatch(/@media[\s\S]*grid-template-areas[\s\S]*"header"[\s\S]*"main"[\s\S]*"left"[\s\S]*"right"[\s\S]*"footer"/i);
  });

  test('includes a breakpoint at 480px', () => {
    expect(css).toMatch(/@media\s*\(max-width\s*:\s*480px\)/i);
  });

  test('hides both sidebars below 480px', () => {
    expect(css).toMatch(/@media[\s\S]*\.sidebar-left,[\s\S]*\.sidebar-right\s*{[^}]*display\s*:\s*none/i);
  });
});
