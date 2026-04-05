const { loadBugfixModule, readBugfixSource } = require('./helpers');
const { processValue } = loadBugfixModule('BF-008');

describe('BF-008: TypeScript Type Narrowing Bug', () => {
  test('string input returns uppercase text', () => {
    expect(processValue('hello')).toBe('HELLO');
  });

  test('number input returns a fixed decimal string', () => {
    expect(processValue(12.345)).toBe('12.35');
  });

  test('solution uses no any types', () => {
    const content = readBugfixSource('BF-008');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});
