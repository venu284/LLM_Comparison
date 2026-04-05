import { filterByStatus, Status, StatusFilter, StatusItem } from '../../solutions/typescript/TS-002';
import { readSolution } from './helpers';

describe('TS-002: Enum and Union Types', () => {
  const items: StatusItem[] = [
    { id: '1', status: Status.Pending },
    { id: '2', status: Status.Active },
    { id: '3', status: Status.Archived }
  ];

  test('enum has the four required values', () => {
    expect(Status.Pending).toBeDefined();
    expect(Status.Active).toBeDefined();
    expect(Status.Inactive).toBeDefined();
    expect(Status.Archived).toBeDefined();
  });

  test('StatusFilter accepts a single status', () => {
    const filter: StatusFilter = Status.Active;
    expect(filter).toBe(Status.Active);
  });

  test('StatusFilter accepts an array of statuses', () => {
    const filter: StatusFilter = [Status.Pending, Status.Archived];
    expect(Array.isArray(filter)).toBe(true);
  });

  test('filterByStatus works with a single status', () => {
    expect(filterByStatus(items, Status.Active)).toEqual([{ id: '2', status: Status.Active }]);
  });

  test('filterByStatus works with multiple statuses', () => {
    expect(filterByStatus(items, [Status.Pending, Status.Archived])).toHaveLength(2);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-002');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

