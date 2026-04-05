import {
  ExtractStrings,
  IsString,
  Nullable,
  pickStrings
} from '../../solutions/typescript/TS-008';
import { readSolution } from './helpers';

describe('TS-008: Mapped Types and Conditional Types', () => {
  test('Nullable makes properties nullable', () => {
    type Example = Nullable<{ name: string; count: number }>;
    const value: Example = {
      name: null,
      count: 3
    };

    expect(value.name).toBeNull();
  });

  test('IsString resolves to true for string', () => {
    const value: IsString<string> = true;
    expect(value).toBe(true);
  });

  test('IsString resolves to false for non-string types', () => {
    const value: IsString<number> = false;
    expect(value).toBe(false);
  });

  test('ExtractStrings keeps only string properties', () => {
    const value: ExtractStrings<{ name: string; count: number; city: string }> = {
      name: 'Ada',
      city: 'London'
    };

    expect(value).toEqual({ name: 'Ada', city: 'London' });
  });

  test('pickStrings returns only string-valued properties', () => {
    expect(
      pickStrings({
        name: 'Ada',
        count: 3,
        active: true,
        city: 'London'
      })
    ).toEqual({
      name: 'Ada',
      city: 'London'
    });
  });

  test('pickStrings works with empty objects', () => {
    expect(pickStrings({})).toEqual({});
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-008');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

