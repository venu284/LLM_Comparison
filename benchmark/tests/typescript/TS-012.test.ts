import {
  DeepPartial,
  DeepReadonly,
  DeepRequired,
  deepMerge
} from '../../solutions/typescript/TS-012';
import { readSolution } from './helpers';

describe('TS-012: Deep Partial and Deep Readonly', () => {
  test('DeepPartial works on nested objects', () => {
    type Example = DeepPartial<{ profile: { name: string; address: { city: string } } }>;
    const value: Example = {
      profile: {
        address: {
          city: 'London'
        }
      }
    };

    expect(value.profile?.address?.city).toBe('London');
  });

  test('DeepReadonly supports nested objects', () => {
    type Example = DeepReadonly<{ profile: { name: string } }>;
    const value: Example = {
      profile: { name: 'Ada' }
    };

    expect(value.profile.name).toBe('Ada');
  });

  test('DeepReadonly uses ReadonlyArray for arrays', () => {
    const content = readSolution('TS-012');
    expect(content).toContain('ReadonlyArray');
  });

  test('DeepRequired removes optional properties recursively', () => {
    type Example = DeepRequired<{ profile?: { name?: string } }>;
    const value: Example = {
      profile: {
        name: 'Ada'
      }
    };

    expect(value.profile.name).toBe('Ada');
  });

  test('deepMerge merges nested objects', () => {
    const result = deepMerge(
      {
        profile: {
          name: 'Ada',
          address: { city: 'London', zip: '1000' }
        },
        active: true
      },
      {
        profile: {
          address: { city: 'Paris' }
        }
      }
    );

    expect(result.profile.address).toEqual({
      city: 'Paris',
      zip: '1000'
    });
  });

  test('deepMerge skips undefined source values', () => {
    const result = deepMerge(
      { name: 'Ada', meta: { active: true } },
      { name: undefined }
    );

    expect(result.name).toBe('Ada');
  });

  test('deepMerge replaces arrays rather than merging them item-by-item', () => {
    const result = deepMerge(
      { tags: ['a', 'b'], meta: { active: true } },
      { tags: ['c'] }
    );

    expect(result.tags).toEqual(['c']);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-012');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

