import { groupBy, pluck, StringRecord } from '../../solutions/typescript/TS-004';
import { readSolution } from './helpers';

describe('TS-004: Array and Record Utility Types', () => {
  test('StringRecord stores string keys and string values', () => {
    const record: StringRecord = {
      a: 'one',
      b: 'two'
    };

    expect(record.a).toBe('one');
  });

  test('groupBy groups items by the given key', () => {
    const result = groupBy(
      [
        { id: '1', role: 'admin' },
        { id: '2', role: 'user' },
        { id: '3', role: 'admin' }
      ],
      'role'
    );

    expect(result.admin).toHaveLength(2);
    expect(result.user).toHaveLength(1);
  });

  test('pluck extracts values for the given key', () => {
    const result = pluck(
      [
        { id: '1', name: 'Ada' },
        { id: '2', name: 'Grace' }
      ],
      'name'
    );

    expect(result).toEqual(['Ada', 'Grace']);
  });

  test('generic typing works with number properties', () => {
    const result = pluck(
      [
        { id: '1', count: 10 },
        { id: '2', count: 20 }
      ],
      'count'
    );

    expect(result).toEqual([10, 20]);
  });

  test('groupBy handles empty arrays', () => {
    const items: Array<{ category: string }> = [];
    expect(groupBy(items, 'category')).toEqual({});
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-004');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});
